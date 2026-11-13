#pragma once
// Unix-socket link to the Python trainer.
//
// Python is the server (it binds SOCK_PATH); the mod connects. Messages are
// newline-delimited JSON in both directions. Everything here is non-blocking
// so a slow or dead Python process can never freeze or crash the game.

#include <optional>
#include <string>

namespace bridge {

inline constexpr char const* SOCK_PATH = "/tmp/gd_ai2.sock";

bool connected();

// Try once to connect. Returns true if a new connection was made.
bool tryConnect();

void disconnect();

// Next complete line from Python (without the '\n'), if one has arrived.
std::optional<std::string> nextLine();

// Block up to timeoutMs waiting for data from Python. True if data is ready.
bool waitReadable(int timeoutMs);

// Queue a line (a '\n' is appended) and try to send it right away.
void send(std::string const& line);

// Push queued output, waiting up to timeoutMs for socket space.
void flush(int timeoutMs);

} // namespace bridge
