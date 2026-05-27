// Copyright © 2025 Apple Inc.

#include <dlfcn.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

#include "jaccl/rdma.h"

#define LOAD_SYMBOL(symbol, variable)                               \
  {                                                                 \
    variable = (decltype(variable))dlsym(librdma_handle_, #symbol); \
    char* error = dlerror();                                        \
    if (error != nullptr) {                                         \
      std::cerr << IBV_TAG << " " << error << std::endl;            \
      librdma_handle_ = nullptr;                                    \
      return;                                                       \
    }                                                               \
  }

namespace {

void* page_aligned_alloc(size_t num_bytes) {
  static size_t page_size = sysconf(_SC_PAGESIZE);
  void* buf;
  if (posix_memalign(&buf, page_size, num_bytes)) {
    return nullptr;
  }
  return buf;
}

const char* errno_name(int value) {
  switch (value) {
    case EBUSY:
      return "EBUSY";
    case ENOMEM:
      return "ENOMEM";
    case ETIMEDOUT:
      return "ETIMEDOUT";
    default:
      return nullptr;
  }
}

void append_errno(std::ostringstream& msg, int value) {
  if (value == 0) {
    return;
  }
  msg << " (errno " << value;
  if (auto name = errno_name(value)) {
    msg << " " << name;
  }
  msg << ")";
}

void append_provider_resource_hint(std::ostringstream& msg, int value) {
  if (value == EBUSY || value == ENOMEM || value == 0) {
    msg << "; hint: AppleThunderboltRDMA provider resources may be exhausted "
        << "or contaminated for this boot; stop live RDMA probes and reboot "
        << "the affected node before retrying";
  }
}

bool bench_trace_enabled() {
  const char* env = std::getenv("JACCL_BENCH_TRACE");
  return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

void bench_trace(const std::string& msg) {
  if (bench_trace_enabled()) {
    std::cerr << IBV_TAG << " trace " << msg << std::endl;
  }
}

} // namespace

namespace jaccl {

IBVWrapper::IBVWrapper() {
  bench_trace("dlopen librdma.dylib start");
  librdma_handle_ = dlopen("librdma.dylib", RTLD_NOW | RTLD_GLOBAL);
  if (librdma_handle_ == nullptr) {
    bench_trace("dlopen librdma.dylib failed");
    return;
  }
  bench_trace("dlopen librdma.dylib done");

  LOAD_SYMBOL(ibv_get_device_list, get_device_list);
  LOAD_SYMBOL(ibv_get_device_name, get_device_name);
  LOAD_SYMBOL(ibv_open_device, open_device);
  LOAD_SYMBOL(ibv_free_device_list, free_device_list);
  LOAD_SYMBOL(ibv_close_device, close_device);

  LOAD_SYMBOL(ibv_alloc_pd, alloc_pd);
  LOAD_SYMBOL(ibv_create_qp, create_qp);
  LOAD_SYMBOL(ibv_create_cq, create_cq);
  LOAD_SYMBOL(ibv_destroy_cq, destroy_cq);
  LOAD_SYMBOL(ibv_destroy_qp, destroy_qp);
  LOAD_SYMBOL(ibv_dealloc_pd, dealloc_pd);

  LOAD_SYMBOL(ibv_query_port, query_port);
  LOAD_SYMBOL(ibv_query_gid, query_gid);
  LOAD_SYMBOL(ibv_modify_qp, modify_qp);
  LOAD_SYMBOL(ibv_reg_mr, reg_mr);
  LOAD_SYMBOL(ibv_dereg_mr, dereg_mr);

  // Not really symbols but leaving them here in case they become symbols in
  // the future.
  //
  // LOAD_SYMBOL(ibv_post_send, post_send);
  // LOAD_SYMBOL(ibv_post_recv, post_recv);
  // LOAD_SYMBOL(ibv_poll_cq, poll_cq);
}

IBVWrapper& ibv() {
  static IBVWrapper wrapper;
  return wrapper;
}

SharedBuffer::SharedBuffer(size_t num_bytes)
    : data_(page_aligned_alloc(num_bytes)), num_bytes_(num_bytes) {}

SharedBuffer::SharedBuffer(SharedBuffer&& b) : data_(nullptr), num_bytes_(0) {
  std::swap(data_, b.data_);
  std::swap(num_bytes_, b.num_bytes_);
  std::swap(memory_regions_, b.memory_regions_);
}

SharedBuffer::~SharedBuffer() {
  for (auto& [pd, mr] : memory_regions_) {
    ibv().dereg_mr(mr);
  }
  if (data_ != nullptr) {
    std::free(data_);
  }
}

void SharedBuffer::register_to_protection_domain(ibv_pd* protection_domain) {
  auto [it, inserted] = memory_regions_.insert({protection_domain, nullptr});
  if (!inserted) {
    throw std::runtime_error(
        "[jaccl] Buffer can be registered once per protection domain");
  }

  it->second = ibv().reg_mr(
      protection_domain,
      data_,
      num_bytes_,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
          IBV_ACCESS_REMOTE_WRITE);
  if (!it->second) {
    throw std::runtime_error("[jaccl] Register memory region failed");
  }
}

Connection::Connection(ibv_context* ctx_)
    : ctx(ctx_),
      protection_domain(nullptr),
      completion_queue(nullptr),
      queue_pair(nullptr) {
  src.local_id = -1;
  src.source_gid_index = -1;
}

Connection::Connection(Connection&& c) : Connection(nullptr) {
  std::swap(ctx, c.ctx);
  std::swap(protection_domain, c.protection_domain);
  std::swap(completion_queue, c.completion_queue);
  std::swap(queue_pair, c.queue_pair);
  std::swap(src, c.src);
}

Connection::~Connection() {
  if (queue_pair != nullptr) {
    ibv().destroy_qp(queue_pair);
  }
  if (completion_queue != nullptr) {
    ibv().destroy_cq(completion_queue);
  }
  if (protection_domain != nullptr) {
    ibv().dealloc_pd(protection_domain);
  }
  if (ctx != nullptr) {
    ibv().close_device(ctx);
  }
}

void Connection::allocate_protection_domain() {
  errno = 0;
  protection_domain = ibv().alloc_pd(ctx);
  if (protection_domain == nullptr) {
    std::ostringstream msg;
    msg << "[jaccl] Couldn't allocate protection domain";
    append_errno(msg, errno);
    append_provider_resource_hint(msg, errno);
    throw std::runtime_error(msg.str());
  }
}

void Connection::create_completion_queue(int num_entries) {
  errno = 0;
  completion_queue = ibv().create_cq(ctx, num_entries, nullptr, nullptr, 0);
  if (completion_queue == nullptr) {
    std::ostringstream msg;
    msg << "[jaccl] Couldn't create completion queue";
    append_errno(msg, errno);
    append_provider_resource_hint(msg, errno);
    throw std::runtime_error(msg.str());
  }
}

void Connection::create_queue_pair() {
  ibv_qp_init_attr init_attr = {};
  init_attr.qp_context = ctx;
  init_attr.send_cq = completion_queue;
  init_attr.recv_cq = completion_queue;
  init_attr.srq = nullptr;
  init_attr.cap.max_send_wr = MAX_SEND_WR;
  init_attr.cap.max_recv_wr = MAX_RECV_WR;
  init_attr.cap.max_send_sge = 1;
  init_attr.cap.max_recv_sge = 1;
  init_attr.cap.max_inline_data = 0;
  init_attr.qp_type = IBV_QPT_UC;
  init_attr.sq_sig_all = 0;

  errno = 0;
  queue_pair = ibv().create_qp(protection_domain, &init_attr);

  if (queue_pair == nullptr) {
    std::ostringstream msg;
    msg << "[jaccl] Couldn't create queue pair";
    append_errno(msg, errno);
    append_provider_resource_hint(msg, errno);
    throw std::runtime_error(msg.str());
  }
}

const Destination& Connection::info() {
  if (queue_pair == nullptr || src.local_id >= 0) {
    return src;
  }

  ibv_port_attr port_attr;
  ibv().query_port(ctx, 1, &port_attr);
  ibv_gid gid = {};
  int source_gid_index = -1;
  for (int i = 0; i < port_attr.gid_tbl_len; i++) {
    ibv_gid tmp;
    if (ibv().query_gid(ctx, 1, i, &tmp) == 0) {
      if (*(uint64_t*)&tmp.raw[0] == 0 && *(uint16_t*)&tmp.raw[8] == 0 &&
          *(uint16_t*)&tmp.raw[10] == 0xffff) {
        gid = tmp;
        source_gid_index = i;
        break;
      }
    }
  }
  if (source_gid_index < 0) {
    throw std::runtime_error("[jaccl] No IPv4-mapped source GID found");
  }

  src.local_id = port_attr.lid;
  src.queue_pair_number = queue_pair->qp_num;
  src.packet_sequence_number = 7;
  src.source_gid_index = source_gid_index;
  src.global_identifier = gid;

  return src;
}

void Connection::queue_pair_init() {
  ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = 1;
  attr.pkey_index = 0;
  attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

  int mask =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

  errno = 0;
  if (int status = ibv().modify_qp(queue_pair, &attr, mask); status != 0) {
    std::ostringstream msg;
    msg << "[jaccl] Changing queue pair to INIT failed";
    append_errno(msg, status);
    append_provider_resource_hint(msg, status);
    throw std::invalid_argument(msg.str());
  }
}

void Connection::queue_pair_rtr(const Destination& dst) {
  ibv_qp_attr attr = {};
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_1024;
  attr.rq_psn = dst.packet_sequence_number;
  attr.dest_qp_num = dst.queue_pair_number;
  attr.ah_attr.dlid = dst.local_id;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = 1;
  attr.ah_attr.is_global = 0;

  if (dst.global_identifier.global.interface_id) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.grh.dgid = dst.global_identifier;
    attr.ah_attr.grh.sgid_index = src.source_gid_index;
  }

  int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
      IBV_QP_RQ_PSN;

  errno = 0;
  if (int status = ibv().modify_qp(queue_pair, &attr, mask); status != 0) {
    std::ostringstream msg;
    msg << "[jaccl] Changing queue pair to RTR failed"
        << " mask=0x" << std::hex << mask << std::dec;
    append_errno(msg, status);
    append_provider_resource_hint(msg, status);
    throw std::invalid_argument(msg.str());
  }
}

void Connection::queue_pair_rts() {
  ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = src.packet_sequence_number;

  int mask = IBV_QP_STATE | IBV_QP_SQ_PSN;

  errno = 0;
  if (int status = ibv().modify_qp(queue_pair, &attr, mask); status != 0) {
    std::ostringstream msg;
    msg << "[jaccl] Changing queue pair to RTS failed";
    append_errno(msg, status);
    append_provider_resource_hint(msg, status);
    throw std::invalid_argument(msg.str());
  }
}

std::vector<Connection> create_connections(
    const std::vector<std::string>& device_names) {
  std::vector<Connection> connections;
  int num_devices = 0;
  bench_trace("get_device_list start");
  ibv_device** devices = ibv().get_device_list(&num_devices);
  bench_trace("get_device_list done count=" + std::to_string(num_devices));
  for (auto& name : device_names) {
    // Empty so add a nullptr context
    if (name.empty()) {
      bench_trace("create_connections skip empty device");
      connections.emplace_back(nullptr);
      continue;
    }

    // Search for the name and try to open the device
    bench_trace("create_connections search device=" + name);
    bool found = false;
    for (int i = 0; i < num_devices; i++) {
      if (name == ibv().get_device_name(devices[i])) {
        found = true;
        bench_trace("open_device start device=" + name);
        auto ctx = ibv().open_device(devices[i]);
        if (ctx == nullptr) {
          std::ostringstream msg;
          msg << "[jaccl] Could not open device " << name;
          throw std::runtime_error(msg.str());
        }
        bench_trace("open_device done device=" + name);
        connections.emplace_back(ctx);
        break;
      }
    }
    if (!found) {
      std::ostringstream msg;
      msg << "[jaccl] RDMA device " << name << " not found";
      throw std::runtime_error(msg.str());
    }
  }
  ibv().free_device_list(devices);

  return connections;
}

SideChannel::SideChannel(int rank, int size, const char* addr)
    : rank_(rank), size_(size) {
  bench_trace(
      "side_channel start rank=" + std::to_string(rank_) +
      " size=" + std::to_string(size_) + " addr=" + std::string(addr));
  auto address = parse_address(addr);

  if (rank_ == 0) {
    TCPSocket server(IBV_TAG);
    bench_trace("side_channel listen start");
    server.listen(IBV_TAG, address);
    bench_trace("side_channel listen done");

    for (int i = 0; i < size - 1; i++) {
      bench_trace("side_channel accept start index=" + std::to_string(i));
      sockets_.push_back(server.accept(IBV_TAG));
      bench_trace("side_channel accept done index=" + std::to_string(i));
    }

    std::vector<int> ranks(size - 1);
    for (int i = 0; i < size - 1; i++) {
      bench_trace("side_channel recv rank start index=" + std::to_string(i));
      sockets_[i].recv(
          IBV_TAG, reinterpret_cast<char*>(&ranks[i]), sizeof(int));
      ranks[i]--;
      bench_trace(
          "side_channel recv rank done index=" + std::to_string(i) +
          " rank=" + std::to_string(ranks[i]));
    }
    for (int i = 0; i < size - 1; i++) {
      while (i != ranks[i]) {
        int target = ranks[i];
        std::swap(sockets_[i], sockets_[target]);
        std::swap(ranks[i], ranks[target]);
      }
    }
    bench_trace("side_channel rank0 done");
  } else {
    bench_trace("side_channel connect start");
    sockets_.push_back(
        TCPSocket::connect(
            IBV_TAG, address, 4, 1000, [](int attempt, int wait) {
              std::cerr << IBV_TAG << " Connection attempt " << attempt
                        << " waiting " << wait << " ms" << std::endl;
            }));
    bench_trace("side_channel connect done");
    bench_trace("side_channel send rank start");
    sockets_[0].send(IBV_TAG, reinterpret_cast<char*>(&rank_), sizeof(int));
    bench_trace("side_channel send rank done");
  }
}

SideChannel::SideChannel(SideChannel&& sc)
    : rank_(sc.rank_), size_(sc.size_), sockets_(std::move(sc.sockets_)) {
  sc.rank_ = -1;
  sc.size_ = -1;
}

} // namespace jaccl
