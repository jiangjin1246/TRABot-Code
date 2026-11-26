#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <boost/json.hpp>
#include <boost/json/src.hpp>
#include <openssl/evp.h>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#ifndef G1_ATOM_NATIVE_REVIEW_STORE_PATH
#error "G1_ATOM_NATIVE_REVIEW_STORE_PATH must be fixed at build time"
#endif

#ifndef G1_ATOM_NATIVE_PREVIEW_ROOT
#error "G1_ATOM_NATIVE_PREVIEW_ROOT must be fixed at build time"
#endif

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr char kStatusSchema[] = "g1_atom_motion_cli_status_v1";
constexpr char kCatalogSchema[] = "semantic_atom_catalog_v1";
constexpr char kOntologySchema[] = "dialogue_function_17_v1";
constexpr char kMotionFormat[] = "g1_joint_motion_v1";
constexpr char kPinnedCatalogSha256[] =
    "6725cb58a85b60a1b885d51a407c1ad7a5abea7d309658b2105cc0d86e264e5b";
constexpr char kPinnedLimitsSha256[] =
    "c83bb9c3fc05890fb13dc2fa8b14fd037e0605195f1792ddee2036b0ace7190b";
constexpr char kPinnedPreviewManifestSha256[] =
    "31b79368c9ef142163901717a596ce580be88b0d4a38c96efe8d520b808a3296";
constexpr char kReviewStorePath[] = G1_ATOM_NATIVE_REVIEW_STORE_PATH;
constexpr char kPreviewRoot[] = G1_ATOM_NATIVE_PREVIEW_ROOT;
constexpr char kTopicArmSdk[] = "rt/arm_sdk";
constexpr char kTopicState[] = "rt/lowstate";
constexpr char kExecutionLockPath[] = "/tmp/robot_cvlab_g1_arm_sdk.lock";
constexpr char kExecutionAbstractSocketName[] = "robot_cvlab_g1_arm_sdk_v1";

constexpr double kControlDt = 0.02;
constexpr double kKp = 60.0;
constexpr double kKd = 1.5;
constexpr double kDq = 0.0;
constexpr double kTau = 0.0;
constexpr double kWeightRampSeconds = 0.5;
constexpr double kInitialStateTimeoutSeconds = 3.0;
constexpr double kStateStaleSeconds = 0.25;
constexpr double kTimingOverrunSeconds = 0.015;
constexpr double kMinimumJerkPeakFactor = 1.875;
constexpr double kMinimumJerkPeakSpeedRadS = 1.0;
constexpr double kMinimumJerkDurationFps = 25.0;
constexpr int kMinimumJerkMinTicks = 2;
constexpr double kMaximumBridgeDurationSeconds = 120.0;
constexpr std::uintmax_t kMaxCatalogBytes = 256ULL * 1024ULL * 1024ULL;
constexpr std::uintmax_t kMaxLimitsBytes = 1024ULL * 1024ULL;
constexpr std::uintmax_t kMaxReviewBytes = 16ULL * 1024ULL * 1024ULL;
constexpr std::uintmax_t kMaxManifestBytes = 16ULL * 1024ULL * 1024ULL;
constexpr std::uintmax_t kMaxPreviewBytes = 512ULL * 1024ULL * 1024ULL;
constexpr std::size_t kJointCount = 14;
constexpr std::size_t kPostureHoldJointCount = 15;
constexpr std::size_t kMaximumSequenceEntryCount = 3;
constexpr std::size_t kMaximumSessionCommandBytes = 64ULL * 1024ULL;
constexpr std::size_t kMaximumSessionIdBytes = 128;
constexpr double kDefaultSessionIdleTimeoutSeconds = 15.0;
constexpr double kMaximumSessionIdleTimeoutSeconds = 3600.0;

std::atomic<bool> g_stop_requested{false};

enum JointIndex {
  kLeftHipPitch,
  kLeftHipRoll,
  kLeftHipYaw,
  kLeftKnee,
  kLeftAnkle,
  kLeftAnkleRoll,
  kRightHipPitch,
  kRightHipRoll,
  kRightHipYaw,
  kRightKnee,
  kRightAnkle,
  kRightAnkleRoll,
  kWaistYaw,
  kWaistRoll,
  kWaistPitch,
  kLeftShoulderPitch,
  kLeftShoulderRoll,
  kLeftShoulderYaw,
  kLeftElbow,
  kLeftWristRoll,
  kLeftWristPitch,
  kLeftWristYaw,
  kRightShoulderPitch,
  kRightShoulderRoll,
  kRightShoulderYaw,
  kRightElbow,
  kRightWristRoll,
  kRightWristPitch,
  kRightWristYaw,
  kNotUsedJoint,
  kNotUsedJoint1,
  kNotUsedJoint2,
  kNotUsedJoint3,
  kNotUsedJoint4,
  kNotUsedJoint5,
};

struct JointMeta {
  const char* name;
  JointIndex index;
};

constexpr std::array<JointMeta, kJointCount> kJoints{{
    {"left_shoulder_pitch", kLeftShoulderPitch},
    {"left_shoulder_roll", kLeftShoulderRoll},
    {"left_shoulder_yaw", kLeftShoulderYaw},
    {"left_elbow", kLeftElbow},
    {"left_wrist_roll", kLeftWristRoll},
    {"left_wrist_pitch", kLeftWristPitch},
    {"left_wrist_yaw", kLeftWristYaw},
    {"right_shoulder_pitch", kRightShoulderPitch},
    {"right_shoulder_roll", kRightShoulderRoll},
    {"right_shoulder_yaw", kRightShoulderYaw},
    {"right_elbow", kRightElbow},
    {"right_wrist_roll", kRightWristRoll},
    {"right_wrist_pitch", kRightWristPitch},
    {"right_wrist_yaw", kRightWristYaw},
}};

// Match the proven presentation runner: the Atom changes only the 14 arm
// joints, while legs and waist are held at the positions measured before
// control acquisition.  This prevents rt/arm_sdk weight blending from turning
// an arm-only Atom into visible whole-body motion.
constexpr std::array<JointMeta, kPostureHoldJointCount> kPostureHoldJoints{{
    {"left_hip_pitch", kLeftHipPitch},
    {"left_hip_roll", kLeftHipRoll},
    {"left_hip_yaw", kLeftHipYaw},
    {"left_knee", kLeftKnee},
    {"left_ankle", kLeftAnkle},
    {"left_ankle_roll", kLeftAnkleRoll},
    {"right_hip_pitch", kRightHipPitch},
    {"right_hip_roll", kRightHipRoll},
    {"right_hip_yaw", kRightHipYaw},
    {"right_knee", kRightKnee},
    {"right_ankle", kRightAnkle},
    {"right_ankle_roll", kRightAnkleRoll},
    {"waist_yaw", kWaistYaw},
    {"waist_roll", kWaistRoll},
    {"waist_pitch", kWaistPitch},
}};

using JointVector = std::array<double, kJointCount>;
using PostureHoldVector = std::array<double, kPostureHoldJointCount>;

struct MeasuredState {
  JointVector arms{};
  PostureHoldVector posture_hold{};
};

class AppError : public std::runtime_error {
 public:
  AppError(std::string code, std::string message)
      : std::runtime_error(std::move(message)), code_(std::move(code)) {}

  const std::string& code() const noexcept { return code_; }

 private:
  std::string code_;
};

class ExecutionLock {
 public:
  ExecutionLock() {
    fd_ = ::open(kExecutionLockPath, O_CREAT | O_RDWR | O_CLOEXEC, 0660);
    if (fd_ < 0) {
      const int error = errno;
      throw AppError("execution_lock_error",
                     std::string("cannot open ") + kExecutionLockPath + ": " +
                         std::strerror(error));
    }
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
      const int error = errno;
      ::close(fd_);
      fd_ = -1;
      if (error == EWOULDBLOCK || error == EAGAIN) {
        throw AppError("execution_lock_busy",
                       std::string("another G1 arm runner holds ") + kExecutionLockPath);
      }
      throw AppError("execution_lock_error",
                     std::string("cannot lock ") + kExecutionLockPath + ": " +
                         std::strerror(error));
    }

    socket_fd_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (socket_fd_ < 0) {
      const int error = errno;
      ReleaseFileLock();
      throw AppError("execution_lock_error",
                     std::string("cannot create abstract UNIX lock socket: ") +
                         std::strerror(error));
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    constexpr std::size_t name_size = sizeof(kExecutionAbstractSocketName) - 1;
    static_assert(name_size + 1 <= sizeof(address.sun_path),
                  "abstract execution lock name is too long");
    address.sun_path[0] = '\0';
    std::memcpy(address.sun_path + 1, kExecutionAbstractSocketName, name_size);
    const auto address_size = static_cast<socklen_t>(
        offsetof(sockaddr_un, sun_path) + 1 + name_size);
    if (::bind(socket_fd_, reinterpret_cast<const sockaddr*>(&address), address_size) != 0) {
      const int error = errno;
      (void)::close(socket_fd_);
      socket_fd_ = -1;
      ReleaseFileLock();
      if (error == EADDRINUSE) {
        throw AppError("execution_lock_busy",
                       "another G1 arm runner owns the abstract UNIX lock");
      }
      throw AppError("execution_lock_error",
                     std::string("cannot bind abstract UNIX lock socket: ") +
                         std::strerror(error));
    }
  }

  ExecutionLock(const ExecutionLock&) = delete;
  ExecutionLock& operator=(const ExecutionLock&) = delete;

  ~ExecutionLock() {
    if (socket_fd_ >= 0) {
      (void)::close(socket_fd_);
    }
    ReleaseFileLock();
  }

 private:
  void ReleaseFileLock() noexcept {
    if (fd_ >= 0) {
      (void)::flock(fd_, LOCK_UN);
      (void)::close(fd_);
      fd_ = -1;
    }
  }

  int fd_ = -1;
  int socket_fd_ = -1;
};

struct MotionFrame {
  double t = 0.0;
  JointVector q{};
};

struct Motion {
  std::string name;
  double duration = 0.0;
  std::vector<MotionFrame> frames;
};

struct CatalogEntry {
  int index = 0;
  std::string entry_id;
  std::string function;
  std::string atom_id;
  std::string version;
  int revision = 0;
  std::string content_sha256;
  std::int64_t duration_us = 0;
  Motion motion;
};

struct Catalog {
  std::string raw_sha256;
  Motion neutral;
  std::int64_t neutral_duration_us = 0;
  JointVector neutral_q{};
  std::vector<CatalogEntry> entries;
};

struct JointLimit {
  double minimum = 0.0;
  double maximum = 0.0;
};

struct Limits {
  std::string raw_sha256;
  std::string source;
  double position_margin = 0.0;
  std::array<JointLimit, kJointCount> joints{};
};

struct LimitsResult {
  double max_abs_q = 0.0;
  double max_segment_velocity = 0.0;
};

struct VisualApproval {
  bool approved = false;
  bool current = false;
  std::string status = "pending";
  std::string review_store_sha256;
  std::string manifest_sha256;
  std::string preview_path;
  std::string preview_sha256;
  std::string reviewer;
  std::string reviewed_at;
  std::string note;
};

struct Options {
  fs::path catalog_path;
  fs::path limits_path;
  std::optional<int> index;
  std::string atom_id;
  std::vector<std::string> entry_ids;
  std::vector<std::string> expect_versions;
  std::vector<int> expect_revisions;
  std::vector<std::string> expect_content_sha256s;
  std::vector<std::string> confirm_entry_ids;
  bool execute = false;
  bool explicit_validate_only = false;
  bool realtime_session = false;
  std::string session_id;
  double session_idle_timeout_seconds = kDefaultSessionIdleTimeoutSeconds;
  bool session_idle_timeout_set = false;
  std::string network;
};

void SignalHandler(int) { g_stop_requested.store(true, std::memory_order_relaxed); }

std::string UsageText(const char* argv0) {
  std::ostringstream out;
  out << "Usage: " << argv0
      << " --catalog=PATH (--entry-id=HEX [--entry-id=HEX ...]|--index=N|--atom-id=ID)"
         " [--expect-version=V ...] [--expect-revision=N ...]"
         " [--expect-content-sha256=HEX ...] [--limits=PATH] [--validate-only]"
         " [--execute --network=IFACE --confirm-entry-id=HEX"
         " [--confirm-entry-id=HEX ...] --limits=PATH]\n       "
      << argv0
      << " --catalog=PATH --limits=PATH --execute --network=IFACE"
         " --realtime-session --session-id=SAFE [--session-idle-timeout=SECONDS]";
  return out.str();
}

std::string JsonString(const boost::json::value& value, const std::string& label) {
  if (!value.is_string()) {
    throw AppError("catalog_validation_error", label + " must be a string");
  }
  const auto& text = value.as_string();
  return std::string(text.data(), text.size());
}

const boost::json::object* JsonObject(const boost::json::value& value,
                                     const std::string& label) {
  if (!value.is_object()) {
    throw AppError("catalog_validation_error", label + " must be an object");
  }
  return &value.as_object();
}

const boost::json::array* JsonArray(const boost::json::value& value,
                                   const std::string& label) {
  if (!value.is_array()) {
    throw AppError("catalog_validation_error", label + " must be an array");
  }
  return &value.as_array();
}

double JsonNumber(const boost::json::value& value, const std::string& label) {
  double result = 0.0;
  if (value.is_double()) {
    result = value.as_double();
  } else if (value.is_int64()) {
    result = static_cast<double>(value.as_int64());
  } else if (value.is_uint64()) {
    result = static_cast<double>(value.as_uint64());
  } else {
    throw AppError("catalog_validation_error", label + " must be numeric");
  }
  if (!std::isfinite(result)) {
    throw AppError("catalog_validation_error", label + " must be finite");
  }
  return result;
}

std::int64_t JsonInteger(const boost::json::value& value, const std::string& label) {
  if (value.is_int64()) {
    return value.as_int64();
  }
  if (value.is_uint64() &&
      value.as_uint64() <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return static_cast<std::int64_t>(value.as_uint64());
  }
  throw AppError("catalog_validation_error", label + " must be an integer");
}

bool JsonBool(const boost::json::value& value, const std::string& label) {
  if (!value.is_bool()) {
    throw AppError("catalog_validation_error", label + " must be boolean");
  }
  return value.as_bool();
}

const boost::json::value& Required(const boost::json::object& object,
                                   const char* key,
                                   const std::string& label) {
  const auto* value = object.if_contains(key);
  if (value == nullptr) {
    throw AppError("catalog_validation_error", label + "." + key + " is required");
  }
  return *value;
}

bool IsLowerHexSha256(const std::string& value) {
  if (value.size() != 64) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

void RequireSha256(const std::string& value, const std::string& label) {
  if (!IsLowerHexSha256(value)) {
    throw AppError("catalog_validation_error", label + " must be a lowercase SHA-256");
  }
}

std::string Sha256Hex(const void* bytes, std::size_t size) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),
                                                                  EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), bytes, size) != 1) {
    throw AppError("sha256_error", "OpenSSL failed to initialize SHA-256");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1 || digest_size != 32) {
    throw AppError("sha256_error", "OpenSSL failed to finalize SHA-256");
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < digest_size; ++i) {
    out << std::setw(2) << static_cast<unsigned int>(digest[i]);
  }
  return out.str();
}

std::string Sha256Hex(const std::string& bytes) { return Sha256Hex(bytes.data(), bytes.size()); }

class UniqueFd {
 public:
  UniqueFd() = default;
  explicit UniqueFd(int fd) : fd_(fd) {}
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      Reset();
      fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
  }
  ~UniqueFd() { Reset(); }

  int get() const noexcept { return fd_; }
  explicit operator bool() const noexcept { return fd_ >= 0; }

 private:
  void Reset() noexcept {
    if (fd_ >= 0) {
      (void)::close(fd_);
      fd_ = -1;
    }
  }

  int fd_ = -1;
};

struct FileSnapshot {
  std::string raw;
  std::string sha256;
};

bool SameFileSnapshot(const struct stat& before, const struct stat& after) {
  if (before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
      before.st_mode != after.st_mode || before.st_size != after.st_size) {
    return false;
  }
#if defined(__APPLE__)
  return before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec &&
         before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec &&
         before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec &&
         before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec;
#else
  return before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
         before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
         before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
         before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
#endif
}

FileSnapshot ReadAndHashRegularFd(int fd,
                                  std::uintmax_t maximum_bytes,
                                  bool keep_raw,
                                  const std::string& error_code,
                                  const std::string& label) {
  struct stat before {};
  if (::fstat(fd, &before) != 0) {
    throw AppError(error_code, "cannot stat " + label + ": " + std::strerror(errno));
  }
  if (!S_ISREG(before.st_mode) || before.st_size < 0 ||
      static_cast<std::uintmax_t>(before.st_size) > maximum_bytes) {
    throw AppError(error_code, label + " must be a bounded regular file");
  }
  if (::lseek(fd, 0, SEEK_SET) < 0) {
    throw AppError(error_code, "cannot seek " + label + ": " + std::strerror(errno));
  }

  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),
                                                                  EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw AppError("sha256_error", "OpenSSL failed to initialize SHA-256 for " + label);
  }
  FileSnapshot result;
  if (keep_raw) {
    result.raw.reserve(static_cast<std::size_t>(before.st_size));
  }
  std::array<char, 1024 * 1024> buffer{};
  std::uintmax_t total = 0;
  for (;;) {
    const ssize_t count = ::read(fd, buffer.data(), buffer.size());
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw AppError(error_code, "cannot read " + label + ": " + std::strerror(errno));
    }
    if (count == 0) {
      break;
    }
    total += static_cast<std::uintmax_t>(count);
    if (total > maximum_bytes) {
      throw AppError(error_code, label + " grew beyond its size limit while being read");
    }
    if (EVP_DigestUpdate(context.get(), buffer.data(), static_cast<std::size_t>(count)) != 1) {
      throw AppError("sha256_error", "OpenSSL failed to hash " + label);
    }
    if (keep_raw) {
      result.raw.append(buffer.data(), static_cast<std::size_t>(count));
    }
  }

  struct stat after {};
  if (::fstat(fd, &after) != 0 || !SameFileSnapshot(before, after) ||
      total != static_cast<std::uintmax_t>(before.st_size)) {
    throw AppError(error_code, label + " changed while it was being verified");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1 || digest_size != 32) {
    throw AppError("sha256_error", "OpenSSL failed to finalize SHA-256 for " + label);
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < digest_size; ++i) {
    out << std::setw(2) << static_cast<unsigned int>(digest[i]);
  }
  result.sha256 = out.str();
  return result;
}

std::optional<FileSnapshot> TryReadRegularSnapshot(const fs::path& path,
                                                    std::uintmax_t maximum_bytes,
                                                    const std::string& error_code,
                                                    const std::string& label) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    if (errno == ENOENT) {
      return std::nullopt;
    }
    throw AppError(error_code, "cannot open " + label + ": " + std::strerror(errno));
  }
  UniqueFd owner(fd);
  return ReadAndHashRegularFd(owner.get(), maximum_bytes, true, error_code, label);
}

UniqueFd OpenPreviewRoot() {
  const int fd = ::open(kPreviewRoot, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_DIRECTORY);
  if (fd < 0) {
    throw AppError("visual_preview_error",
                   std::string("cannot open fixed preview root: ") + std::strerror(errno));
  }
  return UniqueFd(fd);
}

std::vector<std::string> SafeRelativeComponents(const std::string& relative,
                                                const std::string& label) {
  if (relative.empty() || relative.find('\0') != std::string::npos) {
    throw AppError("visual_preview_error", label + " is empty or contains NUL");
  }
  const fs::path path(relative);
  if (path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
    throw AppError("visual_preview_error", label + " must be relative to the fixed preview root");
  }
  std::vector<std::string> components;
  for (const fs::path& component : path) {
    const std::string text = component.string();
    if (text.empty() || text == "." || text == "..") {
      throw AppError("visual_preview_error", label + " contains an unsafe path component");
    }
    components.push_back(text);
  }
  if (components.empty()) {
    throw AppError("visual_preview_error", label + " has no path components");
  }
  return components;
}

UniqueFd OpenRegularBelowPreviewRoot(int root_fd,
                                     const std::string& relative,
                                     const std::string& label) {
  const std::vector<std::string> components = SafeRelativeComponents(relative, label);
  const int duplicated = ::dup(root_fd);
  if (duplicated < 0) {
    throw AppError("visual_preview_error",
                   "cannot duplicate fixed preview root fd: " + std::string(std::strerror(errno)));
  }
  UniqueFd current(duplicated);
  for (std::size_t i = 0; i < components.size(); ++i) {
    const bool final = i + 1 == components.size();
    const int flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW | (final ? 0 : O_DIRECTORY);
    const int next = ::openat(current.get(), components[i].c_str(), flags);
    if (next < 0) {
      throw AppError("visual_preview_error",
                     "cannot open " + label + " below fixed preview root: " +
                         std::strerror(errno));
    }
    current = UniqueFd(next);
  }
  return current;
}

std::string ReadPinnedCatalog(const fs::path& path, std::string& raw_sha256) {
  std::error_code error;
  const fs::file_status status = fs::symlink_status(path, error);
  if (error) {
    throw AppError("catalog_io_error", "cannot stat catalog: " + error.message());
  }
  if (fs::is_symlink(status) || !fs::is_regular_file(status)) {
    throw AppError("catalog_io_error", "catalog must be a regular non-symlink file");
  }
  const auto size = fs::file_size(path, error);
  if (error || size > kMaxCatalogBytes) {
    throw AppError("catalog_io_error", "catalog is unreadable or exceeds 256 MiB");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw AppError("catalog_io_error", "cannot open catalog: " + path.string());
  }
  std::string raw;
  raw.resize(static_cast<std::size_t>(size));
  if (size != 0) {
    input.read(raw.data(), static_cast<std::streamsize>(size));
  }
  if (!input || static_cast<std::uintmax_t>(input.gcount()) != size) {
    throw AppError("catalog_io_error", "catalog changed or ended while being read");
  }
  raw_sha256 = Sha256Hex(raw);
  const std::string expected_name =
      std::string(kCatalogSchema) + "." + raw_sha256 + ".json";
  if (path.filename().string() != expected_name) {
    throw AppError("catalog_identity_error",
                   "catalog filename is not its raw content address; expected " + expected_name);
  }
  if (raw_sha256 != kPinnedCatalogSha256) {
    throw AppError("catalog_identity_error",
                   "catalog SHA-256 is not the pinned 113-Atom release");
  }
  return raw;
}

std::string ReadLimitsFile(const fs::path& path, std::string& raw_sha256) {
  std::error_code error;
  const fs::file_status status = fs::symlink_status(path, error);
  if (error) {
    throw AppError("limits_io_error", "cannot stat limits file: " + error.message());
  }
  if (fs::is_symlink(status) || !fs::is_regular_file(status)) {
    throw AppError("limits_io_error", "limits file must be a regular non-symlink file");
  }
  const auto size = fs::file_size(path, error);
  if (error || size > kMaxLimitsBytes) {
    throw AppError("limits_io_error", "limits file is unreadable or exceeds 1 MiB");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw AppError("limits_io_error", "cannot open limits file: " + path.string());
  }
  std::string raw(static_cast<std::size_t>(size), '\0');
  if (size != 0) {
    input.read(raw.data(), static_cast<std::streamsize>(size));
  }
  if (!input || static_cast<std::uintmax_t>(input.gcount()) != size) {
    throw AppError("limits_io_error", "limits file changed or ended while being read");
  }
  raw_sha256 = Sha256Hex(raw);
  return raw;
}

void ValidateJointNames(const boost::json::value& value, const std::string& label) {
  const auto& joints = *JsonArray(value, label);
  if (joints.size() != kJointCount) {
    throw AppError("catalog_validation_error", label + " must contain exactly 14 joints");
  }
  for (std::size_t i = 0; i < kJointCount; ++i) {
    if (JsonString(joints[i], label + "[" + std::to_string(i) + "]") != kJoints[i].name) {
      throw AppError("catalog_validation_error",
                     label + " does not match the canonical 14-joint order");
    }
  }
}

JointVector ParseQ(const boost::json::value& value, const std::string& label) {
  const auto& object = *JsonObject(value, label);
  if (object.size() != kJointCount) {
    throw AppError("catalog_validation_error", label + " must contain exactly 14 joints");
  }
  JointVector result{};
  for (std::size_t i = 0; i < kJointCount; ++i) {
    result[i] = JsonNumber(Required(object, kJoints[i].name, label),
                           label + "." + kJoints[i].name);
  }
  return result;
}

Motion ParseMotion(const boost::json::value& value,
                   const std::string& label,
                   const std::string* expected_source_hash) {
  const auto& object = *JsonObject(value, label);
  if (JsonString(Required(object, "format", label), label + ".format") != kMotionFormat) {
    throw AppError("catalog_validation_error", label + ".format must equal g1_joint_motion_v1");
  }
  if (JsonString(Required(object, "interpolation", label), label + ".interpolation") !=
      "linear") {
    throw AppError("catalog_validation_error", label + ".interpolation must equal linear");
  }
  ValidateJointNames(Required(object, "joints", label), label + ".joints");
  if (expected_source_hash != nullptr) {
    const std::string source_hash =
        JsonString(Required(object, "source_atom_hash", label), label + ".source_atom_hash");
    RequireSha256(source_hash, label + ".source_atom_hash");
    if (source_hash != *expected_source_hash) {
      throw AppError("catalog_identity_error",
                     label + ".source_atom_hash does not match entry content_sha256");
    }
    if (const auto* parent = object.if_contains("parent_atom_hash");
        parent != nullptr && !parent->is_null()) {
      RequireSha256(JsonString(*parent, label + ".parent_atom_hash"),
                    label + ".parent_atom_hash");
    }
  }

  Motion motion;
  motion.name = JsonString(Required(object, "name", label), label + ".name");
  motion.duration = JsonNumber(Required(object, "duration", label), label + ".duration");
  if (!(motion.duration > 0.0)) {
    throw AppError("catalog_validation_error", label + ".duration must be positive");
  }
  const auto& frames = *JsonArray(Required(object, "frames", label), label + ".frames");
  if (frames.size() < 2) {
    throw AppError("catalog_validation_error", label + ".frames must contain at least two frames");
  }
  motion.frames.reserve(frames.size());
  double previous_t = -1.0;
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const std::string frame_label = label + ".frames[" + std::to_string(i) + "]";
    const auto& frame_object = *JsonObject(frames[i], frame_label);
    MotionFrame frame;
    frame.t = JsonNumber(Required(frame_object, "t", frame_label), frame_label + ".t");
    if (i == 0 && frame.t != 0.0) {
      throw AppError("catalog_validation_error", label + " must start at t=0");
    }
    if (i > 0 && !(frame.t > previous_t)) {
      throw AppError("catalog_validation_error", label + " frame times must be strictly increasing");
    }
    frame.q = ParseQ(Required(frame_object, "q", frame_label), frame_label + ".q");
    motion.frames.push_back(frame);
    previous_t = frame.t;
  }
  if (motion.duration + 1e-9 < motion.frames.back().t ||
      std::abs(motion.duration - motion.frames.back().t) > 1e-9) {
    throw AppError("catalog_validation_error",
                   label + ".duration must equal the final frame time");
  }
  return motion;
}

std::string ExpectedEntryId(const CatalogEntry& entry) {
  // Keys are inserted in Python json.dumps(sort_keys=True) byte order.
  boost::json::object exact;
  exact["atom_id"] = entry.atom_id;
  exact["content_sha256"] = entry.content_sha256;
  exact["function"] = entry.function;
  exact["ontology_version"] = kOntologySchema;
  exact["revision"] = entry.revision;
  exact["version"] = entry.version;
  return Sha256Hex(boost::json::serialize(exact));
}

CatalogEntry ParseEntry(const boost::json::value& value, std::size_t offset) {
  const std::string label = "catalog.entries[" + std::to_string(offset) + "]";
  const auto& object = *JsonObject(value, label);
  CatalogEntry entry;
  const auto index64 = JsonInteger(Required(object, "index", label), label + ".index");
  const auto revision64 =
      JsonInteger(Required(object, "revision", label), label + ".revision");
  entry.duration_us =
      JsonInteger(Required(object, "duration_us", label), label + ".duration_us");
  if (index64 <= 0 || index64 > std::numeric_limits<int>::max() || revision64 <= 0 ||
      revision64 > std::numeric_limits<int>::max() || entry.duration_us <= 0) {
    throw AppError("catalog_validation_error", label + " has invalid positive integers");
  }
  entry.index = static_cast<int>(index64);
  entry.revision = static_cast<int>(revision64);
  entry.entry_id = JsonString(Required(object, "entry_id", label), label + ".entry_id");
  entry.function = JsonString(Required(object, "function", label), label + ".function");
  entry.atom_id = JsonString(Required(object, "atom_id", label), label + ".atom_id");
  entry.version = JsonString(Required(object, "version", label), label + ".version");
  entry.content_sha256 =
      JsonString(Required(object, "content_sha256", label), label + ".content_sha256");
  RequireSha256(entry.entry_id, label + ".entry_id");
  RequireSha256(entry.content_sha256, label + ".content_sha256");
  if (entry.function.empty() || entry.atom_id.empty() || entry.version.empty()) {
    throw AppError("catalog_validation_error", label + " contains an empty identity field");
  }
  if (!JsonBool(Required(object, "approved", label), label + ".approved") ||
      JsonBool(Required(object, "removed", label), label + ".removed")) {
    throw AppError("catalog_validation_error",
                   label + " must be approved=true and removed=false");
  }
  if (ExpectedEntryId(entry) != entry.entry_id) {
    throw AppError("catalog_identity_error", label + ".entry_id does not match exact identity");
  }
  const auto& motion_value = Required(object, "motion", label);
  entry.motion = ParseMotion(motion_value, label + ".motion", &entry.content_sha256);
  if (std::llround(entry.motion.duration * 1'000'000.0) != entry.duration_us) {
    throw AppError("catalog_validation_error", label + ".duration_us does not match motion");
  }
  return entry;
}

Catalog LoadCatalog(const fs::path& path) {
  Catalog catalog;
  const std::string raw = ReadPinnedCatalog(path, catalog.raw_sha256);
  boost::json::value parsed;
  try {
    parsed = boost::json::parse(raw);
  } catch (const std::exception& error) {
    throw AppError("catalog_validation_error",
                   std::string("catalog is not valid JSON: ") + error.what());
  }
  const auto& root = *JsonObject(parsed, "catalog");
  if (JsonString(Required(root, "schema_version", "catalog"), "catalog.schema_version") !=
      kCatalogSchema) {
    throw AppError("catalog_validation_error",
                   "catalog.schema_version must equal semantic_atom_catalog_v1");
  }
  const auto& ontology = *JsonObject(Required(root, "ontology", "catalog"), "catalog.ontology");
  if (JsonString(Required(ontology, "schema_version", "catalog.ontology"),
                 "catalog.ontology.schema_version") != kOntologySchema) {
    throw AppError("catalog_validation_error",
                   "catalog ontology must equal dialogue_function_17_v1");
  }

  const auto& inputs = *JsonObject(Required(root, "inputs", "catalog"), "catalog.inputs");
  const std::string input_neutral_sha =
      JsonString(Required(inputs, "neutral_motion_sha256", "catalog.inputs"),
                 "catalog.inputs.neutral_motion_sha256");
  RequireSha256(input_neutral_sha, "catalog.inputs.neutral_motion_sha256");

  const auto& neutral = *JsonObject(Required(root, "neutral", "catalog"), "catalog.neutral");
  const std::string neutral_sha =
      JsonString(Required(neutral, "motion_sha256", "catalog.neutral"),
                 "catalog.neutral.motion_sha256");
  RequireSha256(neutral_sha, "catalog.neutral.motion_sha256");
  if (neutral_sha != input_neutral_sha) {
    throw AppError("catalog_identity_error",
                   "catalog neutral motion SHA does not match the input binding");
  }
  ValidateJointNames(Required(neutral, "joints", "catalog.neutral"),
                     "catalog.neutral.joints");
  catalog.neutral_q = ParseQ(Required(neutral, "q", "catalog.neutral"), "catalog.neutral.q");
  catalog.neutral_duration_us = JsonInteger(
      Required(neutral, "duration_us", "catalog.neutral"), "catalog.neutral.duration_us");
  if (catalog.neutral_duration_us <= 0) {
    throw AppError("catalog_validation_error", "catalog.neutral.duration_us must be positive");
  }
  catalog.neutral = ParseMotion(Required(neutral, "motion", "catalog.neutral"),
                                "catalog.neutral.motion", nullptr);
  if (std::llround(catalog.neutral.duration * 1'000'000.0) !=
      catalog.neutral_duration_us) {
    throw AppError("catalog_validation_error",
                   "catalog.neutral.duration_us does not match motion");
  }
  for (std::size_t i = 0; i < kJointCount; ++i) {
    if (std::abs(catalog.neutral.frames.front().q[i] - catalog.neutral_q[i]) > 1e-12 ||
        std::abs(catalog.neutral.frames.back().q[i] - catalog.neutral_q[i]) > 1e-12) {
      throw AppError("catalog_validation_error",
                     "catalog neutral endpoints do not match catalog.neutral.q");
    }
  }

  const auto& entries = *JsonArray(Required(root, "entries", "catalog"), "catalog.entries");
  const auto& counts = *JsonObject(Required(root, "counts", "catalog"), "catalog.counts");
  const auto retained =
      JsonInteger(Required(counts, "retained", "catalog.counts"), "catalog.counts.retained");
  if (retained <= 0 || static_cast<std::size_t>(retained) != entries.size()) {
    throw AppError("catalog_validation_error",
                   "catalog.entries length does not match counts.retained");
  }
  catalog.entries.reserve(entries.size());
  std::set<int> indices;
  std::set<std::string> entry_ids;
  std::set<std::string> atom_ids;
  std::set<std::string> exact_ids;
  int previous_index = 0;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    CatalogEntry entry = ParseEntry(entries[i], i);
    const std::string exact = entry.atom_id + "\n" + entry.version + "\n" +
                              std::to_string(entry.revision) + "\n" + entry.content_sha256;
    if (entry.index <= previous_index || !indices.insert(entry.index).second ||
        !entry_ids.insert(entry.entry_id).second || !atom_ids.insert(entry.atom_id).second ||
        !exact_ids.insert(exact).second) {
      throw AppError("catalog_identity_error",
                     "catalog contains unordered or duplicate entry identities");
    }
    previous_index = entry.index;
    catalog.entries.push_back(std::move(entry));
  }
  return catalog;
}

boost::json::value ParseVisualJson(const std::string& raw, const std::string& label) {
  try {
    return boost::json::parse(raw);
  } catch (const std::exception& error) {
    throw AppError("visual_approval_error",
                   label + " is not valid JSON: " + std::string(error.what()));
  }
}

std::optional<std::string> VisualString(const boost::json::object& object,
                                        const char* key) {
  const auto* value = object.if_contains(key);
  if (value == nullptr || !value->is_string()) {
    return std::nullopt;
  }
  const auto& text = value->as_string();
  return std::string(text.data(), text.size());
}

std::optional<std::int64_t> VisualInteger(const boost::json::object& object,
                                          const char* key) {
  const auto* value = object.if_contains(key);
  if (value == nullptr) {
    return std::nullopt;
  }
  if (value->is_int64()) {
    return value->as_int64();
  }
  if (value->is_uint64() &&
      value->as_uint64() <=
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return static_cast<std::int64_t>(value->as_uint64());
  }
  return std::nullopt;
}

VisualApproval InspectVisualApproval(const Catalog& catalog,
                                     const CatalogEntry& entry) {
  VisualApproval result;

  // The manifest and clip are opened relative to a build-time-fixed directory.
  // Every component is O_NOFOLLOW, so the web process cannot redirect native
  // approval to an arbitrary file by changing a CLI argument or symlink.
  UniqueFd preview_root = OpenPreviewRoot();
  UniqueFd manifest_fd =
      OpenRegularBelowPreviewRoot(preview_root.get(), "manifest.json", "preview manifest");
  const FileSnapshot manifest = ReadAndHashRegularFd(
      manifest_fd.get(), kMaxManifestBytes, true, "visual_preview_error", "preview manifest");
  result.manifest_sha256 = manifest.sha256;
  if (manifest.sha256 != kPinnedPreviewManifestSha256) {
    throw AppError("visual_preview_error",
                   "fixed preview manifest does not match the pinned SHA-256");
  }
  const boost::json::value manifest_value =
      ParseVisualJson(manifest.raw, "preview manifest");
  if (!manifest_value.is_object()) {
    throw AppError("visual_preview_error", "preview manifest root must be an object");
  }
  const auto& manifest_root = manifest_value.as_object();
  const auto manifest_schema = VisualString(manifest_root, "schema_version");
  if (!manifest_schema.has_value() ||
      *manifest_schema != "constraint_pass_review_video_v1") {
    throw AppError("visual_preview_error", "preview manifest has an unsupported schema");
  }

  const boost::json::array* rows = nullptr;
  for (const char* key : {"atoms", "entries", "clips"}) {
    const auto* candidate = manifest_root.if_contains(key);
    if (candidate != nullptr && candidate->is_array()) {
      rows = &candidate->as_array();
      break;
    }
  }
  if (rows == nullptr) {
    throw AppError("visual_preview_error",
                   "preview manifest has no atoms, entries, or clips array");
  }

  const boost::json::object* matching_row = nullptr;
  for (const boost::json::value& value : *rows) {
    if (!value.is_object()) {
      continue;
    }
    const auto& row = value.as_object();
    const auto atom_id = VisualString(row, "atom_id");
    const auto version = VisualString(row, "version");
    const auto revision = VisualInteger(row, "revision");
    const auto content_sha256 = VisualString(row, "content_sha256");
    if (!atom_id.has_value() || !version.has_value() || !revision.has_value() ||
        !content_sha256.has_value() || *atom_id != entry.atom_id ||
        *version != entry.version || *revision != entry.revision ||
        *content_sha256 != entry.content_sha256) {
      continue;
    }
    if (matching_row != nullptr) {
      throw AppError("visual_preview_error",
                     "preview manifest contains duplicate exact rows for the selected Atom");
    }
    matching_row = &row;
  }
  if (matching_row == nullptr) {
    throw AppError("visual_preview_error",
                   "preview manifest has no exact row for the selected Atom identity");
  }

  std::optional<std::string> relative_clip;
  for (const char* key : {"clip_file", "video_file", "clip"}) {
    const auto candidate = VisualString(*matching_row, key);
    if (candidate.has_value() && !candidate->empty()) {
      relative_clip = candidate;
      break;
    }
  }
  if (!relative_clip.has_value()) {
    throw AppError("visual_preview_error",
                   "exact preview manifest row has no clip filename");
  }
  if (fs::path(*relative_clip).extension() != ".mp4") {
    throw AppError("visual_preview_error", "exact preview clip must be an MP4 file");
  }
  const auto declared_clip_sha256 = VisualString(*matching_row, "clip_sha256");
  if (!declared_clip_sha256.has_value() || !IsLowerHexSha256(*declared_clip_sha256)) {
    throw AppError("visual_preview_error",
                   "exact preview manifest row has an invalid clip SHA-256");
  }
  UniqueFd clip_fd = OpenRegularBelowPreviewRoot(preview_root.get(), *relative_clip,
                                                 "exact preview clip");
  const FileSnapshot clip = ReadAndHashRegularFd(
      clip_fd.get(), kMaxPreviewBytes, false, "visual_preview_error", "exact preview clip");
  if (clip.sha256 != *declared_clip_sha256) {
    throw AppError("visual_preview_error",
                   "exact preview clip bytes do not match the manifest SHA-256");
  }
  result.preview_path = (fs::path(kPreviewRoot) / fs::path(*relative_clip)).string();
  result.preview_sha256 = clip.sha256;

  const auto review = TryReadRegularSnapshot(fs::path(kReviewStorePath), kMaxReviewBytes,
                                             "visual_approval_error", "visual review store");
  if (!review.has_value()) {
    // Missing means no human decision yet.  It is useful in validate-only
    // mode, but Execute below still requires approved=true.
    result.current = true;
    return result;
  }
  result.review_store_sha256 = review->sha256;
  const boost::json::value review_value = ParseVisualJson(review->raw, "visual review store");
  if (!review_value.is_object()) {
    throw AppError("visual_approval_error", "visual review store root must be an object");
  }
  const auto& review_root = review_value.as_object();
  const auto review_schema = VisualString(review_root, "schema");
  const auto* reviews_value = review_root.if_contains("reviews");
  if (!review_schema.has_value() || *review_schema != "g1_atom_visual_reviews_v1" ||
      reviews_value == nullptr || !reviews_value->is_object()) {
    throw AppError("visual_approval_error", "visual review store has an unsupported schema");
  }
  const auto& reviews = reviews_value->as_object();
  const auto record_iterator = reviews.find(entry.entry_id);
  if (record_iterator == reviews.end() || !record_iterator->value().is_object()) {
    result.current = true;
    return result;
  }
  const auto& record = record_iterator->value().as_object();
  const auto status = VisualString(record, "status");
  result.status = status.value_or("invalid");
  if (const auto value = VisualString(record, "reviewer"); value.has_value()) {
    result.reviewer = *value;
  }
  if (const auto value = VisualString(record, "reviewed_at"); value.has_value()) {
    result.reviewed_at = *value;
  }
  if (const auto value = VisualString(record, "note"); value.has_value()) {
    result.note = *value;
  }
  const auto recorded_motion_sha256 = VisualString(record, "motion_sha256");
  // The raw SHA-256 pin covers every byte of the whole catalog, including the
  // selected motion, while the exact entry tuple binds this review to its row.
  // Re-encoding Boost's parsed floating-point values would not reproduce
  // Python's canonical JSON reliably.  Keep the audit field mandatory and
  // syntactically valid, but rely on the stronger immutable catalog binding.
  result.current = recorded_motion_sha256.has_value() &&
                   IsLowerHexSha256(*recorded_motion_sha256) &&
                   VisualString(record, "catalog_file_sha256") == catalog.raw_sha256 &&
                   VisualString(record, "entry_id") == entry.entry_id &&
                   VisualString(record, "atom_id") == entry.atom_id &&
                   VisualString(record, "version") == entry.version &&
                   VisualInteger(record, "revision") == entry.revision &&
                   VisualString(record, "content_sha256") == entry.content_sha256 &&
                   VisualString(record, "preview_clip_sha256") == result.preview_sha256;
  if (!result.current) {
    result.status = "stale";
  }
  result.approved = result.current && result.status == "approved";
  return result;
}

Limits LoadLimits(const fs::path& path) {
  Limits limits;
  const std::string raw = ReadLimitsFile(path, limits.raw_sha256);
  boost::json::value parsed;
  try {
    parsed = boost::json::parse(raw);
  } catch (const std::exception& error) {
    throw AppError("limits_validation_error",
                   std::string("limits file is not valid JSON: ") + error.what());
  }
  const auto& root = *JsonObject(parsed, "limits");
  if (JsonString(Required(root, "schema", "limits"), "limits.schema") !=
      "g1_atom_limits_v1") {
    throw AppError("limits_validation_error",
                   "limits.schema must equal g1_atom_limits_v1");
  }
  if (const auto* source = root.if_contains("source"); source != nullptr) {
    limits.source = JsonString(*source, "limits.source");
  }
  if (const auto* margin = root.if_contains("position_margin_rad"); margin != nullptr) {
    limits.position_margin = JsonNumber(*margin, "limits.position_margin_rad");
  }
  if (limits.position_margin < 0.0) {
    throw AppError("limits_validation_error",
                   "limits.position_margin_rad must be finite and non-negative");
  }
  if (root.if_contains("max_atom_segment_velocity_rad_s") != nullptr) {
    throw AppError("limits_validation_error",
                   "limits.max_atom_segment_velocity_rad_s is a removed legacy Atom-body gate");
  }

  const auto& joints = *JsonObject(Required(root, "joints", "limits"), "limits.joints");
  if (joints.size() != kJointCount) {
    throw AppError("limits_validation_error",
                   "limits.joints must contain exactly the canonical 14 joints");
  }
  for (std::size_t i = 0; i < kJointCount; ++i) {
    const std::string label = std::string("limits.joints.") + kJoints[i].name;
    const auto& item = *JsonObject(Required(joints, kJoints[i].name, "limits.joints"), label);
    JointLimit limit;
    limit.minimum = JsonNumber(Required(item, "min_rad", label), label + ".min_rad");
    limit.maximum = JsonNumber(Required(item, "max_rad", label), label + ".max_rad");
    if (!(limit.minimum < limit.maximum)) {
      throw AppError("limits_validation_error", label + " must have min_rad < max_rad");
    }
    if (limit.minimum + limits.position_margin > limit.maximum - limits.position_margin) {
      throw AppError("limits_validation_error",
                     label + " has no usable range after position_margin_rad");
    }
    limits.joints[i] = limit;
  }
  return limits;
}

void ValidatePoseAgainstLimits(const JointVector& q,
                               const Limits& limits,
                               const std::string& label) {
  for (std::size_t joint = 0; joint < kJointCount; ++joint) {
    const double allowed_minimum =
        limits.joints[joint].minimum + limits.position_margin;
    const double allowed_maximum =
        limits.joints[joint].maximum - limits.position_margin;
    if (q[joint] < allowed_minimum || q[joint] > allowed_maximum) {
      std::ostringstream message;
      message << label << "." << kJoints[joint].name << "=" << q[joint]
              << " is outside the allowed [" << allowed_minimum << ","
              << allowed_maximum << "] rad range";
      throw AppError("hardware_gate_rejected", message.str());
    }
  }
}

LimitsResult ValidateMotionAgainstLimits(const Motion& motion,
                                         const Limits& limits,
                                         const std::string& label,
                                         bool measure_segment_velocity) {
  LimitsResult result;
  for (std::size_t frame = 0; frame < motion.frames.size(); ++frame) {
    ValidatePoseAgainstLimits(motion.frames[frame].q, limits,
                              label + ".frames[" + std::to_string(frame) + "].q");
    for (double q : motion.frames[frame].q) {
      result.max_abs_q = std::max(result.max_abs_q, std::abs(q));
    }
    if (!measure_segment_velocity || frame == 0) {
      continue;
    }
    const double dt = motion.frames[frame].t - motion.frames[frame - 1].t;
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
      const double velocity =
          std::abs(motion.frames[frame].q[joint] - motion.frames[frame - 1].q[joint]) /
          dt;
      result.max_segment_velocity = std::max(result.max_segment_velocity, velocity);
    }
  }
  return result;
}

int ParsePositiveInt(const std::string& text, const std::string& label) {
  std::size_t consumed = 0;
  long long value = 0;
  try {
    value = std::stoll(text, &consumed, 10);
  } catch (const std::exception&) {
    throw AppError("argument_error", label + " must be a positive integer");
  }
  if (consumed != text.size() || value <= 0 || value > std::numeric_limits<int>::max()) {
    throw AppError("argument_error", label + " must be a positive integer");
  }
  return static_cast<int>(value);
}

double ParsePositiveDouble(const std::string& text,
                           const std::string& label,
                           double maximum) {
  std::size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &consumed);
  } catch (const std::exception&) {
    throw AppError("argument_error", label + " must be a positive finite number");
  }
  if (consumed != text.size() || !std::isfinite(value) || value <= 0.0 ||
      value > maximum) {
    throw AppError("argument_error", label + " must be a positive finite number no greater than " +
                                         std::to_string(maximum));
  }
  return value;
}

bool IsSafeProtocolId(const std::string& value) {
  if (value.empty() || value.size() > kMaximumSessionIdBytes ||
      !std::isalnum(static_cast<unsigned char>(value.front()))) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](unsigned char c) {
    return std::isalnum(c) || c == '_' || c == '-' || c == '.' || c == ':';
  });
}

void RequireSafeProtocolId(const std::string& value, const std::string& label) {
  if (!IsSafeProtocolId(value)) {
    throw AppError("argument_error",
                   label +
                       " must be 1-128 bytes, start with an alphanumeric, and contain only "
                       "ASCII alphanumerics or _ . : -");
  }
}

void RequireOptionalSequenceCardinality(std::size_t actual,
                                        std::size_t expected,
                                        const std::string& label) {
  if (actual != 0 && actual != expected) {
    throw AppError("argument_error",
                   label + " must be omitted or repeated once for every selected Atom");
  }
}

Options ParseArgs(int argc, const char* const argv[]) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto value_after = [&](const char* prefix) {
      return arg.substr(std::string(prefix).size());
    };
    if (arg.rfind("--catalog=", 0) == 0) {
      options.catalog_path = value_after("--catalog=");
    } else if (arg.rfind("--limits=", 0) == 0) {
      options.limits_path = value_after("--limits=");
    } else if (arg.rfind("--index=", 0) == 0) {
      options.index = ParsePositiveInt(value_after("--index="), "--index");
    } else if (arg.rfind("--atom-id=", 0) == 0) {
      options.atom_id = value_after("--atom-id=");
    } else if (arg.rfind("--entry-id=", 0) == 0) {
      options.entry_ids.push_back(value_after("--entry-id="));
    } else if (arg.rfind("--expect-version=", 0) == 0) {
      options.expect_versions.push_back(value_after("--expect-version="));
    } else if (arg.rfind("--expect-revision=", 0) == 0) {
      options.expect_revisions.push_back(
          ParsePositiveInt(value_after("--expect-revision="), "--expect-revision"));
    } else if (arg.rfind("--expect-content-sha256=", 0) == 0) {
      options.expect_content_sha256s.push_back(value_after("--expect-content-sha256="));
    } else if (arg.rfind("--confirm-entry-id=", 0) == 0) {
      options.confirm_entry_ids.push_back(value_after("--confirm-entry-id="));
    } else if (arg.rfind("--network=", 0) == 0) {
      options.network = value_after("--network=");
    } else if (arg == "--realtime-session") {
      options.realtime_session = true;
    } else if (arg.rfind("--session-id=", 0) == 0) {
      options.session_id = value_after("--session-id=");
    } else if (arg.rfind("--session-idle-timeout=", 0) == 0) {
      options.session_idle_timeout_seconds = ParsePositiveDouble(
          value_after("--session-idle-timeout="), "--session-idle-timeout",
          kMaximumSessionIdleTimeoutSeconds);
      options.session_idle_timeout_set = true;
    } else if (arg == "--execute") {
      options.execute = true;
    } else if (arg == "--validate-only") {
      options.explicit_validate_only = true;
    } else if (arg == "--help" || arg == "-h") {
      throw AppError("help", UsageText(argv[0]));
    } else {
      throw AppError("argument_error", "unknown argument: " + arg + "; " + UsageText(argv[0]));
    }
  }
  if (options.catalog_path.empty()) {
    throw AppError("argument_error", "--catalog is required; " + UsageText(argv[0]));
  }
  const int selector_count = static_cast<int>(options.index.has_value()) +
                             static_cast<int>(!options.atom_id.empty()) +
                             static_cast<int>(!options.entry_ids.empty());
  if (options.realtime_session) {
    if (!options.execute || options.explicit_validate_only) {
      throw AppError("argument_error",
                     "--realtime-session requires --execute and cannot use --validate-only");
    }
    if (options.network.empty() || options.limits_path.empty()) {
      throw AppError("argument_error",
                     "--realtime-session requires --network and --limits");
    }
    if (selector_count != 0 || !options.expect_versions.empty() ||
        !options.expect_revisions.empty() ||
        !options.expect_content_sha256s.empty() ||
        !options.confirm_entry_ids.empty()) {
      throw AppError(
          "argument_error",
          "--realtime-session does not accept entry selectors, identity expectations, or "
          "--confirm-entry-id on the command line");
    }
    RequireSafeProtocolId(options.session_id, "--session-id");
    return options;
  }
  if (!options.session_id.empty() || options.session_idle_timeout_set) {
    throw AppError("argument_error",
                   "--session-id and --session-idle-timeout require --realtime-session");
  }
  if (selector_count != 1) {
    throw AppError("argument_error",
                   "exactly one of --entry-id, --index, or --atom-id is required");
  }
  const std::size_t selected_count =
      options.entry_ids.empty() ? 1U : options.entry_ids.size();
  if (selected_count > kMaximumSequenceEntryCount) {
    throw AppError("argument_error", "at most three ordered Atoms may be selected");
  }
  std::set<std::string> requested_entry_ids;
  for (const std::string& entry_id : options.entry_ids) {
    RequireSha256(entry_id, "--entry-id");
    if (!requested_entry_ids.insert(entry_id).second) {
      throw AppError("argument_error", "ordered --entry-id values must not contain duplicates");
    }
  }
  for (const std::string& content_sha256 : options.expect_content_sha256s) {
    RequireSha256(content_sha256, "--expect-content-sha256");
  }
  for (const std::string& confirm_entry_id : options.confirm_entry_ids) {
    RequireSha256(confirm_entry_id, "--confirm-entry-id");
  }
  RequireOptionalSequenceCardinality(options.expect_versions.size(), selected_count,
                                     "--expect-version");
  RequireOptionalSequenceCardinality(options.expect_revisions.size(), selected_count,
                                     "--expect-revision");
  RequireOptionalSequenceCardinality(options.expect_content_sha256s.size(), selected_count,
                                     "--expect-content-sha256");
  if (options.execute && options.explicit_validate_only) {
    throw AppError("argument_error", "--execute and --validate-only are mutually exclusive");
  }
  if (options.execute) {
    if (options.network.empty()) {
      throw AppError("argument_error", "--execute requires --network");
    }
    if (options.confirm_entry_ids.size() != selected_count) {
      throw AppError("argument_error",
                     "--execute requires one ordered --confirm-entry-id for every Atom");
    }
    if (options.limits_path.empty()) {
      throw AppError("argument_error", "--execute requires --limits");
    }
  } else if (!options.confirm_entry_ids.empty()) {
    throw AppError("argument_error", "--confirm-entry-id is only valid with --execute");
  }
  return options;
}

std::vector<const CatalogEntry*> SelectEntries(const Catalog& catalog,
                                               const Options& options) {
  std::vector<const CatalogEntry*> selected_entries;
  const std::size_t selected_count =
      options.entry_ids.empty() ? 1U : options.entry_ids.size();
  selected_entries.reserve(selected_count);

  for (std::size_t position = 0; position < selected_count; ++position) {
    const CatalogEntry* selected = nullptr;
    for (const auto& entry : catalog.entries) {
      const bool match = !options.entry_ids.empty()
                             ? entry.entry_id == options.entry_ids[position]
                             : (options.index.has_value()
                                    ? entry.index == *options.index
                                    : entry.atom_id == options.atom_id);
      if (!match) {
        continue;
      }
      if (selected != nullptr) {
        throw AppError("catalog_identity_error",
                       "selector matched more than one catalog entry");
      }
      selected = &entry;
    }
    if (selected == nullptr) {
      throw AppError("entry_not_found",
                     "selected Atom at sequence position " +
                         std::to_string(position + 1) +
                         " is not present in the pinned catalog");
    }
    if (!options.expect_versions.empty() &&
        selected->version != options.expect_versions[position]) {
      throw AppError("entry_identity_mismatch",
                     "selected Atom version does not match expectation at sequence position " +
                         std::to_string(position + 1));
    }
    if (!options.expect_revisions.empty() &&
        selected->revision != options.expect_revisions[position]) {
      throw AppError("entry_identity_mismatch",
                     "selected Atom revision does not match expectation at sequence position " +
                         std::to_string(position + 1));
    }
    if (!options.expect_content_sha256s.empty() &&
        selected->content_sha256 != options.expect_content_sha256s[position]) {
      throw AppError(
          "entry_identity_mismatch",
          "selected Atom content SHA-256 does not match expectation at sequence position " +
              std::to_string(position + 1));
    }
    if (options.execute && selected->entry_id != options.confirm_entry_ids[position]) {
      throw AppError(
          "entry_identity_mismatch",
          "--confirm-entry-id does not match the selected exact Atom at sequence position " +
              std::to_string(position + 1));
    }
    selected_entries.push_back(selected);
  }
  return selected_entries;
}

boost::json::object EntryJson(const CatalogEntry& entry) {
  boost::json::object value;
  value["index"] = entry.index;
  value["entry_id"] = entry.entry_id;
  value["function"] = entry.function;
  value["atom_id"] = entry.atom_id;
  value["version"] = entry.version;
  value["revision"] = entry.revision;
  value["content_sha256"] = entry.content_sha256;
  value["duration_us"] = entry.duration_us;
  value["frame_count"] = static_cast<std::uint64_t>(entry.motion.frames.size());
  value["interpolation"] = "linear";
  return value;
}

boost::json::object Status(const std::string& event) {
  boost::json::object value;
  value["schema_version"] = kStatusSchema;
  value["event"] = event;
  return value;
}

void Emit(boost::json::object value, bool error = false) {
  std::ostream& stream = error ? std::cerr : std::cout;
  stream << boost::json::serialize(value) << '\n';
  stream.flush();
  if (!error && !stream) {
    throw AppError("status_write_failed", "failed to write an NDJSON status event to stdout");
  }
}

boost::json::object MeasuredQJson(const JointVector& measured) {
  boost::json::object value;
  for (std::size_t i = 0; i < kJointCount; ++i) {
    value[kJoints[i].name] = measured[i];
  }
  return value;
}

JointVector SampleLinear(const Motion& motion, double time) {
  const double t = std::clamp(time, 0.0, motion.duration);
  if (t <= 0.0) {
    return motion.frames.front().q;
  }
  if (t >= motion.duration) {
    return motion.frames.back().q;
  }
  const auto right = std::upper_bound(
      motion.frames.begin(), motion.frames.end(), t,
      [](double target, const MotionFrame& frame) { return target < frame.t; });
  const auto left = std::prev(right);
  const double alpha = (t - left->t) / (right->t - left->t);
  JointVector result{};
  for (std::size_t i = 0; i < kJointCount; ++i) {
    result[i] = left->q[i] + (right->q[i] - left->q[i]) * alpha;
  }
  return result;
}

double MinimumJerkPosition(double value) {
  const double u = std::clamp(value, 0.0, 1.0);
  return 10.0 * u * u * u - 15.0 * u * u * u * u + 6.0 * u * u * u * u * u;
}

double MaximumDelta(const JointVector& from, const JointVector& to) {
  double maximum = 0.0;
  for (std::size_t i = 0; i < kJointCount; ++i) {
    maximum = std::max(maximum, std::abs(to[i] - from[i]));
  }
  return maximum;
}

double MinimumJerkDuration(const JointVector& from, const JointVector& to) {
  const double required =
      kMinimumJerkPeakFactor * MaximumDelta(from, to) / kMinimumJerkPeakSpeedRadS;
  if (!std::isfinite(required) || required > kMaximumBridgeDurationSeconds) {
    throw AppError("bridge_duration_error",
                   "minimum-jerk bridge duration is invalid or exceeds 120 seconds");
  }
  const int ticks = std::max(
      kMinimumJerkMinTicks,
      static_cast<int>(std::ceil(required * kMinimumJerkDurationFps - 1e-12)));
  return ticks / kMinimumJerkDurationFps;
}

JointVector SampleMinimumJerk(const JointVector& from,
                              const JointVector& to,
                              double time,
                              double duration) {
  const double alpha = MinimumJerkPosition(duration <= 0.0 ? 1.0 : time / duration);
  JointVector result{};
  for (std::size_t i = 0; i < kJointCount; ++i) {
    result[i] = from[i] + (to[i] - from[i]) * alpha;
  }
  return result;
}

class StateMonitor {
 public:
  struct Snapshot {
    MeasuredState measured{};
    std::uint64_t generation = 0;
  };

  void OnMessage(const unitree_hg::msg::dds_::LowState_& state) noexcept {
    MeasuredState measured{};
    bool valid = true;
    try {
      for (std::size_t i = 0; i < kJointCount; ++i) {
        measured.arms[i] =
            static_cast<double>(state.motor_state().at(kJoints[i].index).q());
        valid = valid && std::isfinite(measured.arms[i]);
      }
      for (std::size_t i = 0; i < kPostureHoldJointCount; ++i) {
        measured.posture_hold[i] =
            static_cast<double>(state.motor_state().at(kPostureHoldJoints[i].index).q());
        valid = valid && std::isfinite(measured.posture_hold[i]);
      }
    } catch (...) {
      valid = false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    measured_ = measured;
    valid_ = valid;
    received_ = Clock::now();
    ++generation_;
  }

  bool TryFresh(MeasuredState& measured, double maximum_age_seconds) const {
    Snapshot snapshot;
    if (!TryFreshSnapshot(snapshot, maximum_age_seconds)) {
      return false;
    }
    measured = snapshot.measured;
    return true;
  }

  bool TryFreshSnapshot(Snapshot& snapshot, double maximum_age_seconds) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (generation_ == 0 || !valid_ ||
        std::chrono::duration<double>(Clock::now() - received_).count() > maximum_age_seconds) {
      return false;
    }
    snapshot.measured = measured_;
    snapshot.generation = generation_;
    return true;
  }

  bool TryFreshAfter(Snapshot& snapshot,
                     std::uint64_t generation,
                     double maximum_age_seconds) const {
    if (!TryFreshSnapshot(snapshot, maximum_age_seconds)) {
      return false;
    }
    return snapshot.generation > generation;
  }

  Snapshot RequireFreshSnapshot() const {
    Snapshot snapshot;
    if (!TryFreshSnapshot(snapshot, kStateStaleSeconds)) {
      throw AppError("state_stale", "rt/lowstate is missing, invalid, or stale");
    }
    return snapshot;
  }

  MeasuredState RequireFresh() const {
    return RequireFreshSnapshot().measured;
  }

 private:
  mutable std::mutex mutex_;
  MeasuredState measured_{};
  Clock::time_point received_{};
  std::uint64_t generation_ = 0;
  bool valid_ = false;
};

class CommandWriter {
 public:
  struct FailSafeResult {
    bool attempted = false;
    bool released = false;
    std::uint64_t failed_writes = 0;
  };

  explicit CommandWriter(std::shared_ptr<
                         unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>>
                             publisher,
                         PostureHoldVector posture_hold)
      : publisher_(std::move(publisher)), posture_hold_(std::move(posture_hold)) {}

  // Destruction is deliberately side-effect free.  All release writes are
  // explicit so their DDS result can be reported to the caller.
  ~CommandWriter() = default;

  void Publish(const JointVector& q, double weight) {
    const float bounded_weight = static_cast<float>(std::clamp(weight, 0.0, 1.0));
    msg_.motor_cmd().at(kNotUsedJoint).q(bounded_weight);
    for (std::size_t i = 0; i < kJointCount; ++i) {
      auto& command = msg_.motor_cmd().at(kJoints[i].index);
      command.q(static_cast<float>(q[i]));
      command.dq(static_cast<float>(kDq));
      command.kp(static_cast<float>(kKp));
      command.kd(static_cast<float>(kKd));
      command.tau(static_cast<float>(kTau));
    }
    for (std::size_t i = 0; i < kPostureHoldJointCount; ++i) {
      auto& command = msg_.motor_cmd().at(kPostureHoldJoints[i].index);
      command.q(static_cast<float>(posture_hold_[i]));
      command.dq(static_cast<float>(kDq));
      command.kp(static_cast<float>(kKp));
      command.kd(static_cast<float>(kKd));
      command.tau(static_cast<float>(kTau));
    }
    bool written = false;
    try {
      written = publisher_->Write(msg_);
    } catch (const std::exception& error) {
      ++failed_write_count_;
      throw AppError("dds_write_failed",
                     std::string("rt/arm_sdk write threw an exception: ") + error.what());
    } catch (...) {
      ++failed_write_count_;
      throw AppError("dds_write_failed",
                     "rt/arm_sdk write threw a non-standard exception");
    }
    if (!written) {
      ++failed_write_count_;
      throw AppError("dds_write_failed", "rt/arm_sdk write returned false");
    }
    last_q_ = q;
    weight_ = bounded_weight;
    ++command_count_;
  }

  FailSafeResult ReleaseFailSafe() noexcept {
    FailSafeResult result;
    const std::uint64_t failures_before = failed_write_count_;
    if (command_count_ == 0 || weight_ <= 0.0) {
      result.released = weight_ <= 0.0;
      result.failed_writes = failed_write_count_ - failures_before;
      return result;
    }
    result.attempted = true;
    const float initial_weight = weight_;
    const int intervals = static_cast<int>(std::ceil(kWeightRampSeconds / kControlDt));
    const auto start = Clock::now();
    for (int step = 1; step <= intervals; ++step) {
      std::this_thread::sleep_until(start +
                                    std::chrono::duration_cast<Clock::duration>(
                                        std::chrono::duration<double>(step * kControlDt)));
      try {
        const double alpha = static_cast<double>(step) / intervals;
        Publish(last_q_, initial_weight * (1.0 - alpha));
      } catch (...) {
        // Continue toward an explicit final weight-zero attempt even if an
        // intermediate ramp write was rejected by DDS.
      }
    }
    try {
      Publish(last_q_, 0.0);
      result.released = true;
    } catch (...) {
      result.released = false;
    }
    result.failed_writes = failed_write_count_ - failures_before;
    return result;
  }

  float weight() const noexcept { return weight_; }
  const JointVector& last_q() const noexcept { return last_q_; }
  std::uint64_t command_count() const noexcept { return command_count_; }
  std::uint64_t failed_write_count() const noexcept { return failed_write_count_; }

 private:
  std::shared_ptr<unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>>
      publisher_;
  unitree_hg::msg::dds_::LowCmd_ msg_;
  PostureHoldVector posture_hold_{};
  JointVector last_q_{};
  float weight_ = 0.0f;
  std::uint64_t command_count_ = 0;
  std::uint64_t failed_write_count_ = 0;
};

void CheckExecutionHealth(const StateMonitor& monitor,
                          const Clock::time_point& deadline) {
  if (g_stop_requested.load(std::memory_order_relaxed)) {
    throw AppError("interrupted", "SIGINT or SIGTERM requested a fail-safe release");
  }
  (void)monitor.RequireFresh();
  if (Clock::now() > deadline +
                         std::chrono::duration_cast<Clock::duration>(
                             std::chrono::duration<double>(kTimingOverrunSeconds))) {
    throw AppError("timing_overrun",
                   "50 Hz control loop was more than 15 ms late; refusing catch-up writes");
  }
}

JointVector RequireMeasuredAfterPhase(const StateMonitor& monitor,
                                      CommandWriter& writer) {
  const StateMonitor::Snapshot baseline = monitor.RequireFreshSnapshot();
  const int maximum_intervals =
      std::max(1, static_cast<int>(std::ceil(kStateStaleSeconds / kControlDt)));
  const auto start = Clock::now();
  for (int step = 1; step <= maximum_intervals; ++step) {
    StateMonitor::Snapshot next;
    if (monitor.TryFreshAfter(next, baseline.generation, kStateStaleSeconds)) {
      return next.measured.arms;
    }
    const auto deadline =
        start + std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(step * kControlDt));
    std::this_thread::sleep_until(deadline);
    CheckExecutionHealth(monitor, deadline);
    // Keep the exact previous endpoint commanded while waiting for a state
    // generation that is known to be newer than the completed phase.
    writer.Publish(writer.last_q(), 1.0);
    if (monitor.TryFreshAfter(next, baseline.generation, kStateStaleSeconds)) {
      return next.measured.arms;
    }
  }
  throw AppError("state_stale",
                 "no new rt/lowstate generation arrived after the completed phase");
}

template <typename Sampler, typename Weight>
void RunPhase(double duration,
              const StateMonitor& monitor,
              CommandWriter& writer,
              Sampler sample,
              Weight weight) {
  const int full_intervals =
      std::max(0, static_cast<int>(std::floor(duration / kControlDt + 1e-12)));
  const auto start = Clock::now();
  auto publish_at = [&](double scheduled_time) {
    const auto deadline =
        start + std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(scheduled_time));
    if (scheduled_time > 0.0) {
      std::this_thread::sleep_until(deadline);
    }
    CheckExecutionHealth(monitor, deadline);
    const double actual_elapsed =
        scheduled_time == 0.0
            ? 0.0
            : (scheduled_time >= duration
                   ? duration
                   : std::clamp(
                         std::chrono::duration<double>(Clock::now() - start).count(),
                         0.0, duration));
    // Never publish a historical scheduled sample after waking late.  Sampling
    // at real elapsed time keeps commanded displacement proportional to the
    // actual inter-command interval instead of amplifying trajectory speed.
    writer.Publish(sample(actual_elapsed), weight(actual_elapsed, duration));
    return actual_elapsed >= duration;
  };
  for (int step = 0; step <= full_intervals; ++step) {
    if (publish_at(std::min(duration, step * kControlDt))) {
      return;
    }
  }
  const double last_regular_time = full_intervals * kControlDt;
  if (duration - last_regular_time > 1e-9) {
    // Preserve the catalog duration exactly.  A non-20 ms tail is emitted at
    // its true deadline instead of stretching the Atom to the next 50 Hz tick.
    (void)publish_at(duration);
  }
}

enum class SessionCommandKind { kAppend, kEndTurn, kStop, kSnapshot };

struct SessionCommand {
  SessionCommandKind kind = SessionCommandKind::kAppend;
  std::string session_id;
  std::string action_id;
  std::string request_id;
  std::string entry_id;
  std::string expect_version;
  int expect_revision = 0;
  std::string expect_content_sha256;
  std::uint64_t expected_action_seq = 0;
  std::string payload_sha256;
};

std::string ProtocolString(const boost::json::object& object,
                           const char* key) {
  const auto* value = object.if_contains(key);
  if (value == nullptr || !value->is_string()) {
    throw AppError("session_protocol_error",
                   std::string("session command field ") + key + " must be a string");
  }
  const auto& text = value->as_string();
  return std::string(text.data(), text.size());
}

std::uint64_t ProtocolUint64(const boost::json::object& object,
                             const char* key) {
  const auto* value = object.if_contains(key);
  if (value == nullptr) {
    throw AppError("session_protocol_error",
                   std::string("session command field ") + key + " is required");
  }
  if (value->is_uint64()) {
    return value->as_uint64();
  }
  if (value->is_int64() && value->as_int64() >= 0) {
    return static_cast<std::uint64_t>(value->as_int64());
  }
  throw AppError("session_protocol_error",
                 std::string("session command field ") + key +
                     " must be a non-negative integer");
}

int ProtocolPositiveInt(const boost::json::object& object,
                        const char* key) {
  const std::uint64_t value = ProtocolUint64(object, key);
  if (value == 0 || value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw AppError("session_protocol_error",
                   std::string("session command field ") + key +
                       " must be a positive 32-bit integer");
  }
  return static_cast<int>(value);
}

void RequireExactProtocolFields(const boost::json::object& object,
                                std::initializer_list<const char*> fields,
                                const std::string& command_type) {
  if (object.size() != fields.size()) {
    throw AppError("session_protocol_error",
                   command_type + " command has missing, duplicate, or unknown fields");
  }
  for (const char* field : fields) {
    if (object.if_contains(field) == nullptr) {
      throw AppError("session_protocol_error",
                     command_type + " command is missing field " + field);
    }
  }
}

void RequireSafeSessionField(const std::string& value,
                             const std::string& label) {
  if (!IsSafeProtocolId(value)) {
    throw AppError("session_protocol_error",
                   label +
                       " must be 1-128 bytes, start with an alphanumeric, and contain only "
                       "ASCII alphanumerics or _ . : -");
  }
}

std::size_t CountTopLevelObjectMembers(const std::string& line) {
  std::size_t object_depth = 0;
  std::size_t array_depth = 0;
  std::size_t member_count = 0;
  bool in_string = false;
  bool escaped = false;
  for (unsigned char byte : line) {
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (byte == '\\') {
        escaped = true;
      } else if (byte == '"') {
        in_string = false;
      }
      continue;
    }
    if (byte == '"') {
      in_string = true;
    } else if (byte == '{') {
      ++object_depth;
    } else if (byte == '}') {
      if (object_depth > 0) {
        --object_depth;
      }
    } else if (byte == '[') {
      ++array_depth;
    } else if (byte == ']') {
      if (array_depth > 0) {
        --array_depth;
      }
    } else if (byte == ':' && object_depth == 1 && array_depth == 0) {
      ++member_count;
    }
  }
  return member_count;
}

SessionCommand ParseSessionCommandLine(const std::string& line) {
  if (line.empty()) {
    throw AppError("session_protocol_error", "empty NDJSON command line");
  }
  boost::json::value parsed;
  try {
    parsed = boost::json::parse(line);
  } catch (const std::exception& error) {
    throw AppError("session_protocol_error",
                   std::string("invalid NDJSON command: ") + error.what());
  }
  if (!parsed.is_object()) {
    throw AppError("session_protocol_error", "session command must be a JSON object");
  }
  const auto& object = parsed.as_object();
  if (CountTopLevelObjectMembers(line) != object.size()) {
    throw AppError("session_protocol_error",
                   "session command must not contain duplicate object fields");
  }
  const std::string type = ProtocolString(object, "type");
  SessionCommand command;
  if (type == "append") {
    RequireExactProtocolFields(
        object,
        {"type", "session_id", "action_id", "entry_id", "expect_version",
         "expect_revision", "expect_content_sha256", "expected_action_seq",
         "payload_sha256"},
        type);
    command.kind = SessionCommandKind::kAppend;
    command.entry_id = ProtocolString(object, "entry_id");
    command.expect_version = ProtocolString(object, "expect_version");
    command.expect_revision = ProtocolPositiveInt(object, "expect_revision");
    command.expect_content_sha256 = ProtocolString(object, "expect_content_sha256");
    if (!IsLowerHexSha256(command.entry_id) ||
        !IsLowerHexSha256(command.expect_content_sha256)) {
      throw AppError("session_protocol_error",
                     "append entry_id and expect_content_sha256 must be lowercase SHA-256");
    }
    if (command.expect_version.empty() || command.expect_version.size() > 128) {
      throw AppError("session_protocol_error",
                     "append expect_version must contain 1-128 bytes");
    }
  } else if (type == "end_turn") {
    RequireExactProtocolFields(
        object,
        {"type", "session_id", "action_id", "expected_action_seq",
         "payload_sha256"},
        type);
    command.kind = SessionCommandKind::kEndTurn;
  } else if (type == "stop") {
    RequireExactProtocolFields(
        object,
        {"type", "session_id", "action_id", "expected_action_seq",
         "payload_sha256"},
        type);
    command.kind = SessionCommandKind::kStop;
  } else if (type == "snapshot") {
    RequireExactProtocolFields(
        object,
        {"type", "session_id", "request_id", "expected_action_seq"},
        type);
    command.kind = SessionCommandKind::kSnapshot;
    command.request_id = ProtocolString(object, "request_id");
    RequireSafeSessionField(command.request_id, "request_id");
  } else {
    throw AppError("session_protocol_error",
                   "unsupported session command type: " + type);
  }
  command.session_id = ProtocolString(object, "session_id");
  command.expected_action_seq = ProtocolUint64(object, "expected_action_seq");
  RequireSafeSessionField(command.session_id, "session_id");
  if (command.kind != SessionCommandKind::kSnapshot) {
    command.action_id = ProtocolString(object, "action_id");
    command.payload_sha256 = ProtocolString(object, "payload_sha256");
    RequireSafeSessionField(command.action_id, "action_id");
    if (!IsLowerHexSha256(command.payload_sha256)) {
      throw AppError("session_protocol_error", "payload_sha256 must be a lowercase SHA-256");
    }
  }
  return command;
}

class SessionCommandReader {
 public:
  std::string ReadLineWhileHolding(const StateMonitor& monitor,
                                   CommandWriter& writer,
                                   const Clock::time_point& idle_started,
                                   double idle_timeout_seconds) {
    const auto idle_deadline =
        idle_started + std::chrono::duration_cast<Clock::duration>(
                           std::chrono::duration<double>(idle_timeout_seconds));
    auto next_control_deadline =
        Clock::now() + std::chrono::duration_cast<Clock::duration>(
                           std::chrono::duration<double>(kControlDt));
    for (;;) {
      if (g_stop_requested.load(std::memory_order_relaxed)) {
        throw AppError("interrupted", "SIGINT or SIGTERM requested a fail-safe release");
      }

      const auto now = Clock::now();
      if (now >= idle_deadline) {
        throw AppError("session_idle_timeout",
                       "realtime session reached its idle timeout");
      }
      if (now >= next_control_deadline) {
        CheckExecutionHealth(monitor, next_control_deadline);
        writer.Publish(writer.last_q(), 1.0);
        next_control_deadline += std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(kControlDt));
        continue;
      }

      if (const auto line = ExtractLine(); line.has_value()) {
        return *line;
      }

      const auto wake_at = std::min(next_control_deadline, idle_deadline);
      const double wait_seconds =
          std::max(0.0, std::chrono::duration<double>(wake_at - Clock::now()).count());
      const int wait_ms = std::max(
          0, static_cast<int>(std::ceil(wait_seconds * 1000.0 - 1e-12)));
      struct pollfd descriptor {};
      descriptor.fd = STDIN_FILENO;
      descriptor.events = POLLIN | POLLHUP;
      const int result = ::poll(&descriptor, 1, wait_ms);
      if (result < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw AppError("session_read_failed",
                       std::string("poll(stdin) failed: ") + std::strerror(errno));
      }
      if (result == 0) {
        continue;
      }
      if ((descriptor.revents & (POLLERR | POLLNVAL)) != 0) {
        throw AppError("session_read_failed", "stdin reported POLLERR or POLLNVAL");
      }
      if ((descriptor.revents & (POLLIN | POLLHUP)) != 0) {
        std::array<char, 4096> bytes{};
        const ssize_t count = ::read(STDIN_FILENO, bytes.data(), bytes.size());
        if (count < 0) {
          if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
            continue;
          }
          throw AppError("session_read_failed",
                         std::string("read(stdin) failed: ") + std::strerror(errno));
        }
        if (count == 0) {
          throw AppError("session_input_eof",
                         "stdin closed before an end_turn command completed the session");
        }
        buffer_.append(bytes.data(), static_cast<std::size_t>(count));
        EnforceLineLimit();
      }
    }
  }

 private:
  std::optional<std::string> ExtractLine() {
    const std::size_t newline = buffer_.find('\n');
    if (newline == std::string::npos) {
      return std::nullopt;
    }
    if (newline > kMaximumSessionCommandBytes) {
      throw AppError("session_protocol_error", "NDJSON command line exceeds 64 KiB");
    }
    std::string line = buffer_.substr(0, newline);
    buffer_.erase(0, newline + 1);
    if (!line.empty() && line.back() == '\r') {
      throw AppError("session_protocol_error", "NDJSON commands must use LF, not CRLF");
    }
    EnforceLineLimit();
    return line;
  }

  void EnforceLineLimit() const {
    const std::size_t newline = buffer_.find('\n');
    const std::size_t first_line_bytes =
        newline == std::string::npos ? buffer_.size() : newline;
    if (first_line_bytes > kMaximumSessionCommandBytes) {
      throw AppError("session_protocol_error", "NDJSON command line exceeds 64 KiB");
    }
  }

  std::string buffer_;
};

template <typename Result>
Result WaitForGateWhileHolding(std::future<Result>& future,
                               const StateMonitor& monitor,
                               CommandWriter& writer) {
  auto next_control_deadline =
      Clock::now() + std::chrono::duration_cast<Clock::duration>(
                         std::chrono::duration<double>(kControlDt));
  try {
    while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      std::this_thread::sleep_until(next_control_deadline);
      CheckExecutionHealth(monitor, next_control_deadline);
      writer.Publish(writer.last_q(), 1.0);
      next_control_deadline += std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>(kControlDt));
    }
  } catch (...) {
    // A std::launch::async future may wait for its worker in its destructor.
    // Release control before stack unwinding can wait on filesystem validation.
    (void)writer.ReleaseFailSafe();
    throw;
  }
  return future.get();
}

boost::json::object SessionStatus(const std::string& event,
                                  const Options& options,
                                  const std::string& action_id,
                                  std::uint64_t action_seq,
                                  const std::string& payload_sha256) {
  auto status = Status(event);
  status["session_id"] = options.session_id;
  status["action_id"] = action_id;
  status["action_seq"] = action_seq;
  status["payload_sha256"] = payload_sha256;
  return status;
}

const CatalogEntry& FindRealtimeEntry(const Catalog& catalog,
                                      const SessionCommand& command) {
  const CatalogEntry* selected = nullptr;
  for (const CatalogEntry& candidate : catalog.entries) {
    if (candidate.entry_id == command.entry_id) {
      if (selected != nullptr) {
        throw AppError("catalog_identity_error",
                       "realtime entry_id matched more than one catalog row");
      }
      selected = &candidate;
    }
  }
  if (selected == nullptr) {
    throw AppError("entry_not_found",
                   "realtime append entry_id is absent from the pinned catalog");
  }
  if (selected->version != command.expect_version ||
      selected->revision != command.expect_revision ||
      selected->content_sha256 != command.expect_content_sha256) {
    throw AppError("entry_identity_mismatch",
                   "realtime append exact Atom identity does not match the pinned catalog");
  }
  return *selected;
}

boost::json::array EntryIdsJson(
    const std::vector<const CatalogEntry*>& entries) {
  boost::json::array result;
  result.reserve(entries.size());
  for (const CatalogEntry* entry : entries) {
    result.emplace_back(entry->entry_id);
  }
  return result;
}

void AddSequenceContext(boost::json::object& status,
                        const CatalogEntry& entry,
                        std::size_t position,
                        std::size_t count) {
  status["entry_id"] = entry.entry_id;
  status["sequence_position"] = static_cast<std::uint64_t>(position + 1);
  status["sequence_count"] = static_cast<std::uint64_t>(count);
}

void EmitPhase(const std::string& event,
               const CatalogEntry& entry,
               std::size_t position,
               std::size_t count,
               double duration,
               const std::string& policy) {
  auto status = Status(event);
  AddSequenceContext(status, entry, position, count);
  status["duration_seconds"] = duration;
  status["policy"] = policy;
  Emit(std::move(status));
}

void ExecuteOnRobot(const Options& options,
                    const Catalog& catalog,
                    const std::vector<const CatalogEntry*>& entries,
                    const Limits& limits) {
  if (entries.empty() || entries.size() > kMaximumSequenceEntryCount) {
    throw AppError("execution_error", "execution requires one to three ordered Atoms");
  }
  const CatalogEntry& first_entry = *entries.front();
  const std::size_t sequence_count = entries.size();
  ExecutionLock execution_lock;
  (void)execution_lock;
  auto lock_status = Status("execution_lock_acquired");
  lock_status["entry_id"] = first_entry.entry_id;
  lock_status["entry_ids"] = EntryIdsJson(entries);
  lock_status["sequence_count"] = static_cast<std::uint64_t>(sequence_count);
  lock_status["path"] = kExecutionLockPath;
  lock_status["abstract_socket"] =
      std::string("@") + kExecutionAbstractSocketName;
  lock_status["pid"] = static_cast<std::int64_t>(::getpid());
  Emit(std::move(lock_status));

  g_stop_requested.store(false, std::memory_order_relaxed);
  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  unitree::robot::ChannelFactory::Instance()->Init(0, options.network);
  auto publisher =
      std::make_shared<unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>>(
          kTopicArmSdk);
  publisher->InitChannel();

  StateMonitor monitor;
  auto subscriber =
      std::make_shared<unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>>(
          kTopicState);
  subscriber->InitChannel(
      [&](const void* message) {
        const auto* state = static_cast<const unitree_hg::msg::dds_::LowState_*>(message);
        if (state != nullptr) {
          monitor.OnMessage(*state);
        }
      },
      1);

  auto waiting = Status("waiting_for_state");
  waiting["entry_id"] = first_entry.entry_id;
  waiting["entry_ids"] = EntryIdsJson(entries);
  waiting["sequence_count"] = static_cast<std::uint64_t>(sequence_count);
  waiting["network"] = options.network;
  waiting["timeout_seconds"] = kInitialStateTimeoutSeconds;
  Emit(std::move(waiting));

  MeasuredState measured{};
  const auto state_deadline = Clock::now() +
                              std::chrono::duration_cast<Clock::duration>(
                                  std::chrono::duration<double>(kInitialStateTimeoutSeconds));
  while (!monitor.TryFresh(measured, kStateStaleSeconds)) {
    if (g_stop_requested.load(std::memory_order_relaxed)) {
      throw AppError("interrupted", "SIGINT or SIGTERM received before control acquisition");
    }
    if (Clock::now() >= state_deadline) {
      throw AppError("state_timeout", "timed out waiting for a fresh rt/lowstate sample");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ValidatePoseAgainstLimits(measured.arms, limits, "measured_q_before_control");

  CommandWriter writer(publisher, measured.posture_hold);
  bool completed = false;
  std::size_t current_position = 0;
  const CatalogEntry* current_entry = entries.front();
  try {
    auto posture_hold = Status("posture_hold_captured");
    posture_hold["entry_id"] = first_entry.entry_id;
    posture_hold["entry_ids"] = EntryIdsJson(entries);
    posture_hold["sequence_count"] = static_cast<std::uint64_t>(sequence_count);
    posture_hold["joint_count"] =
        static_cast<std::uint64_t>(kPostureHoldJointCount);
    posture_hold["source"] = "fresh_rt_lowstate_before_control";
    posture_hold["policy"] = "hold_measured_q";
    Emit(std::move(posture_hold));

    auto acquired = Status("control_weight_ramp_started");
    acquired["entry_id"] = first_entry.entry_id;
    acquired["entry_ids"] = EntryIdsJson(entries);
    acquired["sequence_count"] = static_cast<std::uint64_t>(sequence_count);
    acquired["duration_seconds"] = kWeightRampSeconds;
    Emit(std::move(acquired));
    RunPhase(
        kWeightRampSeconds, monitor, writer,
        [initial_arms = measured.arms](double) { return initial_arms; },
        [](double time, double duration) { return duration <= 0.0 ? 1.0 : time / duration; });

    for (std::size_t position = 0; position < sequence_count; ++position) {
      current_position = position;
      current_entry = entries[position];
      const CatalogEntry& entry = *current_entry;
      const JointVector bridge_origin =
          position == 0 ? monitor.RequireFresh().arms
                        : RequireMeasuredAfterPhase(monitor, writer);
      ValidatePoseAgainstLimits(
          bridge_origin, limits,
          "measured_q_before_entry_bridge[" + std::to_string(position) + "]");
      const JointVector atom_first = entry.motion.frames.front().q;
      const double entry_bridge_duration = MinimumJerkDuration(bridge_origin, atom_first);
      auto bridge =
          Status(position == 0 ? "entry_bridge_started" : "inter_entry_bridge_started");
      AddSequenceContext(bridge, entry, position, sequence_count);
      bridge["duration_seconds"] = entry_bridge_duration;
      bridge["policy"] = "minimum_jerk_v1";
      bridge["origin"] = position == 0
                             ? "fresh_measured_q_after_control_acquisition"
                             : "new_lowstate_generation_after_previous_atom";
      if (position > 0) {
        const CatalogEntry& previous = *entries[position - 1];
        bridge["previous_entry_id"] = previous.entry_id;
        bridge["previous_atom_endpoint_tracking_error_rad"] =
            MaximumDelta(bridge_origin, previous.motion.frames.back().q);
      }
      Emit(std::move(bridge));
      RunPhase(
          entry_bridge_duration, monitor, writer,
          [&](double time) {
            return SampleMinimumJerk(bridge_origin, atom_first, time,
                                     entry_bridge_duration);
          },
          [](double, double) { return 1.0; });

      EmitPhase("atom_started", entry, position, sequence_count,
                entry.motion.duration, "linear");
      RunPhase(entry.motion.duration, monitor, writer,
               [&](double time) { return SampleLinear(entry.motion, time); },
               [](double, double) { return 1.0; });
    }

    current_position = sequence_count - 1;
    current_entry = entries.back();
    const CatalogEntry& final_entry = *current_entry;
    const JointVector atom_last = final_entry.motion.frames.back().q;
    const JointVector recovery_origin = RequireMeasuredAfterPhase(monitor, writer);
    ValidatePoseAgainstLimits(recovery_origin, limits, "measured_q_before_recovery_bridge");
    const double tracking_error = MaximumDelta(recovery_origin, atom_last);
    const JointVector neutral_first = catalog.neutral.frames.front().q;
    const double recovery_bridge_duration =
        MinimumJerkDuration(recovery_origin, neutral_first);
    auto recovery = Status("recovery_bridge_started");
    AddSequenceContext(recovery, final_entry, current_position, sequence_count);
    recovery["duration_seconds"] = recovery_bridge_duration;
    recovery["policy"] = "minimum_jerk_v1";
    recovery["origin"] = "new_lowstate_generation_after_final_atom";
    recovery["atom_endpoint_tracking_error_rad"] = tracking_error;
    Emit(std::move(recovery));
    RunPhase(
        recovery_bridge_duration, monitor, writer,
        [&](double time) {
          return SampleMinimumJerk(recovery_origin, neutral_first, time,
                                   recovery_bridge_duration);
        },
        [](double, double) { return 1.0; });

    EmitPhase("neutral_started", final_entry, current_position, sequence_count,
              catalog.neutral.duration, "linear");
    RunPhase(catalog.neutral.duration, monitor, writer,
             [&](double time) { return SampleLinear(catalog.neutral, time); },
             [](double, double) { return 1.0; });

    auto releasing = Status("control_weight_release_started");
    AddSequenceContext(releasing, final_entry, current_position, sequence_count);
    releasing["duration_seconds"] = kWeightRampSeconds;
    Emit(std::move(releasing));
    const JointVector neutral_last = catalog.neutral.frames.back().q;
    RunPhase(
        kWeightRampSeconds, monitor, writer, [neutral_last](double) { return neutral_last; },
        [](double time, double duration) {
          return duration <= 0.0 ? 0.0 : 1.0 - time / duration;
        });
    writer.Publish(neutral_last, 0.0);
    if (writer.weight() != 0.0f) {
      throw AppError("dds_release_failed",
                     "final successful DDS command did not record weight=0");
    }
    completed = true;

    auto done = Status("completed");
    AddSequenceContext(done, final_entry, current_position, sequence_count);
    done["entry_ids"] = EntryIdsJson(entries);
    done["command_count"] = writer.command_count();
    done["released"] = writer.weight() == 0.0f;
    done["final_weight"] = writer.weight();
    done["failed_dds_writes"] = writer.failed_write_count();
    Emit(std::move(done));
  } catch (...) {
    const CommandWriter::FailSafeResult release = writer.ReleaseFailSafe();
    auto fail_safe = Status("fail_safe_release");
    AddSequenceContext(fail_safe, *current_entry, current_position, sequence_count);
    fail_safe["entry_ids"] = EntryIdsJson(entries);
    fail_safe["release_attempted"] = release.attempted;
    fail_safe["released"] = release.released;
    fail_safe["final_weight"] = writer.weight();
    fail_safe["failed_dds_writes"] = release.failed_writes;
    fail_safe["total_failed_dds_writes"] = writer.failed_write_count();
    fail_safe["command_count"] = writer.command_count();
    Emit(std::move(fail_safe), true);
    throw;
  }
  if (!completed) {
    throw AppError("execution_error", "execution did not reach completion");
  }
}

void ExecuteRealtimeSession(const Options& options,
                            const Catalog& catalog,
                            const Limits& limits) {
  ExecutionLock execution_lock;
  (void)execution_lock;
  auto lock_status = SessionStatus("execution_lock_acquired", options, "", 0, "");
  lock_status["path"] = kExecutionLockPath;
  lock_status["abstract_socket"] =
      std::string("@") + kExecutionAbstractSocketName;
  lock_status["pid"] = static_cast<std::int64_t>(::getpid());
  Emit(std::move(lock_status));

  g_stop_requested.store(false, std::memory_order_relaxed);
  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  unitree::robot::ChannelFactory::Instance()->Init(0, options.network);
  auto publisher =
      std::make_shared<unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>>(
          kTopicArmSdk);
  publisher->InitChannel();

  StateMonitor monitor;
  auto subscriber =
      std::make_shared<unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>>(
          kTopicState);
  subscriber->InitChannel(
      [&](const void* message) {
        const auto* state = static_cast<const unitree_hg::msg::dds_::LowState_*>(message);
        if (state != nullptr) {
          monitor.OnMessage(*state);
        }
      },
      1);

  auto waiting = SessionStatus("waiting_for_state", options, "", 0, "");
  waiting["network"] = options.network;
  waiting["timeout_seconds"] = kInitialStateTimeoutSeconds;
  Emit(std::move(waiting));

  MeasuredState measured{};
  const auto state_deadline = Clock::now() +
                              std::chrono::duration_cast<Clock::duration>(
                                  std::chrono::duration<double>(kInitialStateTimeoutSeconds));
  while (!monitor.TryFresh(measured, kStateStaleSeconds)) {
    if (g_stop_requested.load(std::memory_order_relaxed)) {
      throw AppError("interrupted", "SIGINT or SIGTERM received before control acquisition");
    }
    if (Clock::now() >= state_deadline) {
      throw AppError("state_timeout", "timed out waiting for a fresh rt/lowstate sample");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ValidatePoseAgainstLimits(measured.arms, limits, "measured_q_before_control");

  CommandWriter writer(publisher, measured.posture_hold);
  std::uint64_t action_seq = 0;
  std::string current_action_id;
  std::string current_payload_sha256;
  std::string current_entry_id;
  try {
    auto posture_hold = SessionStatus("posture_hold_captured", options, "", 0, "");
    posture_hold["joint_count"] =
        static_cast<std::uint64_t>(kPostureHoldJointCount);
    posture_hold["source"] = "fresh_rt_lowstate_before_control";
    posture_hold["policy"] = "hold_measured_q";
    Emit(std::move(posture_hold));

    auto acquired =
        SessionStatus("control_weight_ramp_started", options, "", 0, "");
    acquired["duration_seconds"] = kWeightRampSeconds;
    Emit(std::move(acquired));
    RunPhase(
        kWeightRampSeconds, monitor, writer,
        [initial_arms = measured.arms](double) { return initial_arms; },
        [](double time, double duration) { return duration <= 0.0 ? 1.0 : time / duration; });

    JointVector ready_measured = monitor.RequireFresh().arms;
    ValidatePoseAgainstLimits(ready_measured, limits, "measured_q_at_session_ready");
    auto ready = SessionStatus("session_ready", options, "", action_seq, "");
    ready["measured_q"] = MeasuredQJson(ready_measured);
    ready["catalog_sha256"] = catalog.raw_sha256;
    ready["limits_sha256"] = limits.raw_sha256;
    ready["control_hz"] = 1.0 / kControlDt;
    ready["bridge_policy"] = "minimum_jerk_v1";
    ready["idle_timeout_seconds"] = options.session_idle_timeout_seconds;
    Emit(std::move(ready));

    SessionCommandReader reader;
    std::unordered_set<std::string> accepted_action_ids;
    Clock::time_point idle_started = Clock::now();
    for (;;) {
      const std::string line = reader.ReadLineWhileHolding(
          monitor, writer, idle_started, options.session_idle_timeout_seconds);
      SessionCommand command = ParseSessionCommandLine(line);
      if (command.kind != SessionCommandKind::kSnapshot) {
        current_action_id = command.action_id;
        current_payload_sha256 = command.payload_sha256;
        current_entry_id = command.entry_id;
      }
      if (command.session_id != options.session_id) {
        throw AppError("session_id_mismatch",
                       "session command does not target the active session_id");
      }
      if (command.expected_action_seq != action_seq) {
        throw AppError("action_seq_mismatch",
                       "session command expected_action_seq does not match current action_seq");
      }
      if (command.kind == SessionCommandKind::kSnapshot) {
        const JointVector snapshot_measured = monitor.RequireFresh().arms;
        ValidatePoseAgainstLimits(snapshot_measured, limits,
                                  "measured_q_at_state_snapshot");
        auto snapshot = Status("state_snapshot");
        snapshot["session_id"] = options.session_id;
        snapshot["request_id"] = command.request_id;
        snapshot["action_seq"] = action_seq;
        snapshot["measured_q"] = MeasuredQJson(snapshot_measured);
        Emit(std::move(snapshot));
        idle_started = Clock::now();
        continue;
      }
      if (accepted_action_ids.find(command.action_id) != accepted_action_ids.end()) {
        throw AppError("duplicate_action_id",
                       "session action_id was already accepted in this session");
      }

      if (command.kind == SessionCommandKind::kStop) {
        accepted_action_ids.insert(command.action_id);
        auto stopping = SessionStatus("stop_accepted", options, command.action_id,
                                      action_seq, command.payload_sha256);
        stopping["expected_action_seq"] = command.expected_action_seq;
        Emit(std::move(stopping));
        throw AppError("session_stopped",
                       "stop command requested an immediate fail-safe release");
      }

      if (command.kind == SessionCommandKind::kEndTurn) {
        if (action_seq == std::numeric_limits<std::uint64_t>::max()) {
          throw AppError("action_seq_overflow", "realtime action_seq cannot be incremented");
        }
        accepted_action_ids.insert(command.action_id);
        ++action_seq;
        auto recovery_accepted = SessionStatus(
            "action_accepted", options, command.action_id, action_seq,
            command.payload_sha256);
        recovery_accepted["kind"] = "recovery";
        recovery_accepted["expected_action_seq"] = command.expected_action_seq;
        Emit(std::move(recovery_accepted));

        const JointVector recovery_origin = monitor.RequireFresh().arms;
        ValidatePoseAgainstLimits(recovery_origin, limits,
                                  "measured_q_before_session_recovery");
        const JointVector neutral_first = catalog.neutral.frames.front().q;
        const double recovery_bridge_duration =
            MinimumJerkDuration(recovery_origin, neutral_first);
        auto recovery_started = SessionStatus(
            "action_started", options, command.action_id, action_seq,
            command.payload_sha256);
        recovery_started["kind"] = "recovery";
        recovery_started["expected_action_seq"] = command.expected_action_seq;
        recovery_started["bridge_duration_seconds"] = recovery_bridge_duration;
        recovery_started["neutral_duration_seconds"] = catalog.neutral.duration;
        recovery_started["bridge_policy"] = "minimum_jerk_v1";
        recovery_started["origin"] = "fresh_measured_q_after_last_semantic_action";
        Emit(std::move(recovery_started));

        auto recovery = SessionStatus("recovery_started", options, command.action_id,
                                      action_seq, command.payload_sha256);
        recovery["kind"] = "recovery";
        recovery["expected_action_seq"] = command.expected_action_seq;
        recovery["bridge_duration_seconds"] = recovery_bridge_duration;
        recovery["bridge_policy"] = "minimum_jerk_v1";
        recovery["origin"] = "fresh_measured_q_after_last_semantic_action";
        Emit(std::move(recovery));

        RunPhase(
            recovery_bridge_duration, monitor, writer,
            [&](double time) {
              return SampleMinimumJerk(recovery_origin, neutral_first, time,
                                       recovery_bridge_duration);
            },
            [](double, double) { return 1.0; });
        RunPhase(catalog.neutral.duration, monitor, writer,
                 [&](double time) { return SampleLinear(catalog.neutral, time); },
                 [](double, double) { return 1.0; });

        const JointVector neutral_last = catalog.neutral.frames.back().q;
        auto releasing = SessionStatus("control_weight_release_started", options,
                                       command.action_id, action_seq,
                                       command.payload_sha256);
        releasing["duration_seconds"] = kWeightRampSeconds;
        Emit(std::move(releasing));
        RunPhase(
            kWeightRampSeconds, monitor, writer,
            [neutral_last](double) { return neutral_last; },
            [](double time, double duration) {
              return duration <= 0.0 ? 0.0 : 1.0 - time / duration;
            });
        writer.Publish(neutral_last, 0.0);
        if (writer.weight() != 0.0f) {
          throw AppError("dds_release_failed",
                         "final successful DDS command did not record weight=0");
        }
        const JointVector final_measured = monitor.RequireFresh().arms;

        auto recovery_done = SessionStatus("action_completed", options,
                                           command.action_id, action_seq,
                                           command.payload_sha256);
        recovery_done["kind"] = "recovery";
        recovery_done["measured_q"] = MeasuredQJson(final_measured);
        recovery_done["released"] = true;
        Emit(std::move(recovery_done));

        auto done = SessionStatus("completed", options, command.action_id,
                                  action_seq, command.payload_sha256);
        done["kind"] = "recovery";
        done["command_count"] = writer.command_count();
        done["released"] = true;
        done["final_weight"] = writer.weight();
        done["failed_dds_writes"] = writer.failed_write_count();
        done["measured_q"] = MeasuredQJson(final_measured);
        Emit(std::move(done));
        return;
      }

      const CatalogEntry& entry = FindRealtimeEntry(catalog, command);
      auto gate_future = std::async(std::launch::async, [&catalog, &entry, &limits]() {
        (void)ValidateMotionAgainstLimits(entry.motion, limits, "realtime_atom", true);
        return InspectVisualApproval(catalog, entry);
      });
      const VisualApproval approval =
          WaitForGateWhileHolding(gate_future, monitor, writer);
      if (!approval.approved) {
        throw AppError(
            "visual_approval_required",
            "realtime append requires a current approved native visual-review record");
      }
      if (action_seq == std::numeric_limits<std::uint64_t>::max()) {
        throw AppError("action_seq_overflow", "realtime action_seq cannot be incremented");
      }
      accepted_action_ids.insert(command.action_id);
      ++action_seq;

      auto accepted = SessionStatus("action_accepted", options, command.action_id,
                                    action_seq, command.payload_sha256);
      accepted["kind"] = "semantic";
      accepted["entry_id"] = entry.entry_id;
      accepted["expected_action_seq"] = command.expected_action_seq;
      accepted["entry"] = EntryJson(entry);
      accepted["visual_review_store_sha256"] = approval.review_store_sha256;
      Emit(std::move(accepted));

      const JointVector bridge_origin = monitor.RequireFresh().arms;
      ValidatePoseAgainstLimits(bridge_origin, limits,
                                "measured_q_before_realtime_entry_bridge");
      const JointVector atom_first = entry.motion.frames.front().q;
      const double bridge_duration = MinimumJerkDuration(bridge_origin, atom_first);
      auto started = SessionStatus("action_started", options, command.action_id,
                                   action_seq, command.payload_sha256);
      started["kind"] = "semantic";
      started["entry_id"] = entry.entry_id;
      started["bridge_duration_seconds"] = bridge_duration;
      started["atom_duration_seconds"] = entry.motion.duration;
      started["bridge_policy"] = "minimum_jerk_v1";
      started["origin"] = "fresh_measured_q";
      Emit(std::move(started));

      RunPhase(
          bridge_duration, monitor, writer,
          [&](double time) {
            return SampleMinimumJerk(bridge_origin, atom_first, time,
                                     bridge_duration);
          },
          [](double, double) { return 1.0; });
      RunPhase(entry.motion.duration, monitor, writer,
               [&](double time) { return SampleLinear(entry.motion, time); },
               [](double, double) { return 1.0; });

      const JointVector completed_measured = RequireMeasuredAfterPhase(monitor, writer);
      ValidatePoseAgainstLimits(completed_measured, limits,
                                "measured_q_after_realtime_atom");
      auto action_done = SessionStatus("action_completed", options, command.action_id,
                                       action_seq, command.payload_sha256);
      action_done["kind"] = "semantic";
      action_done["entry_id"] = entry.entry_id;
      action_done["measured_q"] = MeasuredQJson(completed_measured);
      Emit(std::move(action_done));
      idle_started = Clock::now();
    }
  } catch (...) {
    const CommandWriter::FailSafeResult release = writer.ReleaseFailSafe();
    auto fail_safe = SessionStatus("fail_safe_release", options, current_action_id,
                                   action_seq, current_payload_sha256);
    if (!current_entry_id.empty()) {
      fail_safe["entry_id"] = current_entry_id;
    }
    fail_safe["release_attempted"] = release.attempted;
    fail_safe["released"] = release.released;
    fail_safe["final_weight"] = writer.weight();
    fail_safe["failed_dds_writes"] = release.failed_writes;
    fail_safe["total_failed_dds_writes"] = writer.failed_write_count();
    fail_safe["command_count"] = writer.command_count();
    Emit(std::move(fail_safe), true);
    throw;
  }
}

int ExitCodeFor(const std::string& code) {
  if (code == "help") {
    return 0;
  }
  if (code == "argument_error") {
    return 2;
  }
  if (code.find("catalog_") == 0 || code.find("limits_") == 0 ||
      code.find("visual_") == 0 || code.find("hardware_gate_") == 0 ||
      code == "sha256_error" ||
      code == "entry_not_found" || code == "entry_identity_mismatch") {
    return 3;
  }
  if (code == "state_timeout") {
    return 4;
  }
  return 5;
}

}  // namespace

int main(int argc, const char* const argv[]) {
  // A closed status pipe must become a catchable write failure so an active
  // controller can run its explicit fail-safe release instead of dying on
  // SIGPIPE before stack unwinding.
  std::signal(SIGPIPE, SIG_IGN);
  try {
    const Options options = ParseArgs(argc, argv);
    const Catalog catalog = LoadCatalog(options.catalog_path);
    if (options.realtime_session) {
      const Limits limits = LoadLimits(options.limits_path);
      if (limits.raw_sha256 != kPinnedLimitsSha256) {
        throw AppError("limits_identity_error",
                       "realtime execute requires the pinned g1_limits.json raw SHA-256");
      }
      (void)ValidateMotionAgainstLimits(catalog.neutral, limits, "neutral", false);

      auto validated = SessionStatus("validated", options, "", 0, "");
      validated["mode"] = "realtime_session";
      validated["will_send_commands"] = true;
      validated["catalog_path"] = options.catalog_path.string();
      validated["catalog_sha256"] = catalog.raw_sha256;
      validated["catalog_entry_count"] =
          static_cast<std::uint64_t>(catalog.entries.size());
      validated["limits_path"] = options.limits_path.string();
      validated["limits_sha256"] = limits.raw_sha256;
      validated["control_hz"] = 1.0 / kControlDt;
      validated["bridge_policy"] = "minimum_jerk_v1";
      validated["atom_controlled_joint_count"] =
          static_cast<std::uint64_t>(kJointCount);
      validated["posture_hold_joint_count"] =
          static_cast<std::uint64_t>(kPostureHoldJointCount);
      validated["visual_approval_policy"] = "revalidate_exact_identity_before_each_append";
      Emit(std::move(validated));

      ExecuteRealtimeSession(options, catalog, limits);
      return 0;
    }
    const std::vector<const CatalogEntry*> entries = SelectEntries(catalog, options);
    std::optional<Limits> limits;
    std::vector<LimitsResult> limits_results;
    if (!options.limits_path.empty()) {
      limits = LoadLimits(options.limits_path);
      if (options.execute && limits->raw_sha256 != kPinnedLimitsSha256) {
        throw AppError("limits_identity_error",
                       "execute requires the pinned g1_limits.json raw SHA-256");
      }
      limits_results.reserve(entries.size());
      for (std::size_t position = 0; position < entries.size(); ++position) {
        limits_results.push_back(ValidateMotionAgainstLimits(
            entries[position]->motion, *limits,
            "atoms[" + std::to_string(position) + "]", true));
      }
      (void)ValidateMotionAgainstLimits(catalog.neutral, *limits, "neutral", false);
    }

    std::vector<VisualApproval> visual_approvals;
    visual_approvals.reserve(entries.size());
    std::optional<std::string> review_store_sha256;
    for (std::size_t position = 0; position < entries.size(); ++position) {
      VisualApproval approval = InspectVisualApproval(catalog, *entries[position]);
      if (review_store_sha256.has_value() &&
          *review_store_sha256 != approval.review_store_sha256) {
        throw AppError("visual_approval_error",
                       "visual review store changed while validating the Atom sequence");
      }
      review_store_sha256 = approval.review_store_sha256;
      if (options.execute && !approval.approved) {
        throw AppError(
            "visual_approval_required",
            "execute requires a current approved native visual-review record at sequence "
            "position " +
                std::to_string(position + 1));
      }
      visual_approvals.push_back(std::move(approval));
    }

    auto validated = Status("validated");
    validated["mode"] = options.execute ? "execute" : "validate_only";
    validated["will_send_commands"] = options.execute;
    validated["catalog_path"] = options.catalog_path.string();
    validated["catalog_sha256"] = catalog.raw_sha256;
    validated["catalog_entry_count"] = static_cast<std::uint64_t>(catalog.entries.size());
    validated["sequence_count"] = static_cast<std::uint64_t>(entries.size());
    validated["entry_ids"] = EntryIdsJson(entries);
    boost::json::array entry_values;
    entry_values.reserve(entries.size());
    for (const CatalogEntry* entry : entries) {
      entry_values.emplace_back(EntryJson(*entry));
    }
    validated["entries"] = std::move(entry_values);
    if (entries.size() == 1) {
      validated["entry"] = EntryJson(*entries.front());
    }
    boost::json::object neutral;
    neutral["duration_us"] = catalog.neutral_duration_us;
    neutral["frame_count"] = static_cast<std::uint64_t>(catalog.neutral.frames.size());
    neutral["interpolation"] = "linear";
    validated["neutral"] = std::move(neutral);
    validated["control_hz"] = 1.0 / kControlDt;
    validated["bridge_policy"] = "minimum_jerk_v1";
    validated["atom_controlled_joint_count"] =
        static_cast<std::uint64_t>(kJointCount);
    validated["posture_hold_joint_count"] =
        static_cast<std::uint64_t>(kPostureHoldJointCount);
    validated["posture_hold_policy"] = "hold_measured_q_from_control_acquisition";
    validated["visual_approval_checked"] = true;
    boost::json::array visual_gates;
    visual_gates.reserve(entries.size());
    for (std::size_t position = 0; position < entries.size(); ++position) {
      const VisualApproval& approval = visual_approvals[position];
      boost::json::object visual_gate;
      AddSequenceContext(visual_gate, *entries[position], position, entries.size());
      visual_gate["review_store_path"] = kReviewStorePath;
      visual_gate["review_store_sha256"] = approval.review_store_sha256;
      visual_gate["preview_root"] = kPreviewRoot;
      visual_gate["manifest_sha256"] = approval.manifest_sha256;
      visual_gate["execute_pinned_manifest_sha256"] = kPinnedPreviewManifestSha256;
      visual_gate["matches_execute_manifest_pin"] =
          approval.manifest_sha256 == kPinnedPreviewManifestSha256;
      visual_gate["preview_path"] = approval.preview_path;
      visual_gate["preview_sha256"] = approval.preview_sha256;
      visual_gate["status"] = approval.status;
      visual_gate["current"] = approval.current;
      visual_gate["approved"] = approval.approved;
      visual_gate["reviewer"] = approval.reviewer;
      visual_gate["reviewed_at"] = approval.reviewed_at;
      visual_gate["note"] = approval.note;
      if (entries.size() == 1) {
        validated["visual_gate"] = visual_gate;
      }
      visual_gates.emplace_back(std::move(visual_gate));
    }
    validated["visual_gates"] = std::move(visual_gates);
    validated["hardware_gate_checked"] = limits.has_value();
    if (limits.has_value()) {
      boost::json::array hardware_gates;
      hardware_gates.reserve(entries.size());
      for (std::size_t position = 0; position < entries.size(); ++position) {
        const LimitsResult& result = limits_results[position];
        boost::json::object gate;
        AddSequenceContext(gate, *entries[position], position, entries.size());
        gate["limits_path"] = options.limits_path.string();
        gate["limits_sha256"] = limits->raw_sha256;
        gate["limits_source"] = limits->source;
        gate["execute_pinned_sha256"] = kPinnedLimitsSha256;
        gate["matches_execute_pin"] = limits->raw_sha256 == kPinnedLimitsSha256;
        gate["position_margin_rad"] = limits->position_margin;
        gate["atom_segment_velocity_gate_enabled"] = false;
        gate["atom_max_abs_q_rad"] = result.max_abs_q;
        gate["atom_max_segment_velocity_rad_s"] = result.max_segment_velocity;
        gate["passed"] = true;
        if (entries.size() == 1) {
          validated["hardware_gate"] = gate;
        }
        hardware_gates.emplace_back(std::move(gate));
      }
      validated["hardware_gates"] = std::move(hardware_gates);
    }
    Emit(std::move(validated));

    if (!options.execute) {
      auto done = Status("validation_complete");
      done["entry_id"] = entries.front()->entry_id;
      done["entry_ids"] = EntryIdsJson(entries);
      done["sequence_count"] = static_cast<std::uint64_t>(entries.size());
      done["sent_commands"] = false;
      Emit(std::move(done));
      return 0;
    }

    if (!limits.has_value()) {
      throw AppError("hardware_gate_required", "--execute requires a validated limits file");
    }
    ExecuteOnRobot(options, catalog, entries, *limits);
    return 0;
  } catch (const AppError& error) {
    auto status = Status(error.code() == "help" ? "help" : "error");
    status["code"] = error.code();
    status["message"] = error.what();
    Emit(std::move(status), error.code() != "help");
    return ExitCodeFor(error.code());
  } catch (const std::exception& error) {
    auto status = Status("error");
    status["code"] = "unhandled_exception";
    status["message"] = error.what();
    Emit(std::move(status), true);
    return 6;
  } catch (...) {
    auto status = Status("error");
    status["code"] = "unknown_exception";
    status["message"] = "unknown non-standard exception";
    Emit(std::move(status), true);
    return 7;
  }
}
