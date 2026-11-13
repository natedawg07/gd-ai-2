#include "bridge.hpp"

#include <Geode/Geode.hpp>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace geode::prelude;

namespace bridge {
namespace {

int g_fd = -1;
std::string g_in;   // bytes received but not yet split into lines
std::string g_out;  // bytes queued but not yet accepted by the kernel

void closeFd() {
    if (g_fd >= 0) close(g_fd);
    g_fd = -1;
    g_in.clear();
    g_out.clear();
}

// Read everything currently available without blocking.
void pump() {
    if (g_fd < 0) return;
    char buf[4096];
    while (true) {
        ssize_t n = recv(g_fd, buf, sizeof(buf), 0);
        if (n > 0) {
            g_in.append(buf, static_cast<size_t>(n));
            continue;
        }
        if (n == 0) {  // Python closed the connection
            log::info("[bridge] trainer disconnected");
            closeFd();
            return;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) return;
        if (errno == EINTR) continue;
        log::warn("[bridge] recv error: {}", std::strerror(errno));
        closeFd();
        return;
    }
}

// Send as much queued output as the kernel will take right now.
// Unlike the old mod, a full buffer never drops or corrupts a message:
// the rest just stays queued.
bool pushOut() {
    while (g_fd >= 0 && !g_out.empty()) {
        ssize_t n = ::send(g_fd, g_out.data(), g_out.size(), 0);
        if (n > 0) {
            g_out.erase(0, static_cast<size_t>(n));
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return false;
        if (n < 0 && errno == EINTR) continue;
        log::warn("[bridge] send error: {}", std::strerror(errno));
        closeFd();
        return false;
    }
    return true;
}

} // namespace

bool connected() { return g_fd >= 0; }

bool tryConnect() {
    if (g_fd >= 0) return false;

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path) - 1);

    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(fd);
        return false;
    }

    // A write to a closed socket must not raise SIGPIPE (that would kill GD).
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
    // Big send buffer so the level message usually goes out in one call.
    int sz = 1 << 20;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);

    g_fd = fd;
    g_in.clear();
    g_out.clear();
    log::info("[bridge] connected to trainer at {}", SOCK_PATH);
    return true;
}

void disconnect() { closeFd(); }

std::optional<std::string> nextLine() {
    pump();
    auto pos = g_in.find('\n');
    if (pos == std::string::npos) return std::nullopt;
    std::string line = g_in.substr(0, pos);
    g_in.erase(0, pos + 1);
    return line;
}

bool waitReadable(int timeoutMs) {
    if (g_fd < 0) return false;
    if (g_in.find('\n') != std::string::npos) return true;
    pollfd p{g_fd, POLLIN, 0};
    return poll(&p, 1, timeoutMs) > 0;
}

void send(std::string const& line) {
    if (g_fd < 0) return;
    g_out += line;
    g_out += '\n';
    pushOut();
}

void flush(int timeoutMs) {
    if (g_fd < 0) return;
    if (pushOut()) return;
    if (timeoutMs <= 0) return;
    pollfd p{g_fd, POLLOUT, 0};
    while (g_fd >= 0 && !g_out.empty() && poll(&p, 1, timeoutMs) > 0) {
        if (pushOut()) return;
    }
}

} // namespace bridge
