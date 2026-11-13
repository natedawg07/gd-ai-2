// GD AI 2 Bridge: turns a PlayLayer into a lockstep RL environment.
//
// While the Python trainer is connected (and "Agent control" is on), the
// game does NOT advance on its own. Each frame we look for commands:
//
//   {"cmd":"step","a":0|1|2}  -> apply the action, run FRAME_SKIP fixed
//                                physics updates, reply with one state line
//   {"cmd":"reset"}           -> restart the attempt, reply with a state line
//                                (preceded by a one-time "level" message)
//
// Actions: 0 = nothing (button up), 1 = click (down for the first update,
// up for the second), 2 = hold (down for every update).
//
// Inputs go straight into the game through handleButton() and resets call
// resetLevel(), so no keyboard, window focus, or sleeps are involved.

#include <Geode/Geode.hpp>
#include <Geode/modify/AppDelegate.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <chrono>
#include <cstdio>
#include <string>

#include "bridge.hpp"

using namespace geode::prelude;

namespace {

constexpr int    FRAME_SKIP       = 2;            // physics updates per agent step
constexpr float  FIXED_DT         = 1.f / 60.f;   // dt fed to every update
constexpr int    CONNECT_EVERY    = 30;           // frames between connect attempts
constexpr int    FRAME_BUDGET_MS  = 12;           // max time spent serving per frame
constexpr int    LEVEL_FLUSH_MS   = 2000;         // level message may be large
constexpr int    BUTTON_JUMP      = 1;

enum Action { NOTHING = 0, CLICK = 1, HOLD = 2 };

// Per-connection state.
struct Session {
    PlayLayer* layer = nullptr;
    bool levelSent = false;
    bool held = false;    // is our jump button currently down?
    bool dead = false;    // a real death happened since the last reset
    bool won = false;     // level complete since the last reset
} g;

int g_frame = 0;

bool agentEnabled() { return Mod::get()->getSettingValue<bool>("agent-control"); }
bool agentActive() { return agentEnabled() && bridge::connected(); }

void setButton(GJBaseGameLayer* layer, bool down) {
    if (g.held == down) return;
    layer->handleButton(down, BUTTON_JUMP, true);
    g.held = down;
}

// The trainer went away mid-attempt: release the button and, if the player
// is lying dead (we blocked GD's auto-restart), restart so you can play.
void handBack(PlayLayer* pl) {
    setButton(pl, false);
    bool dead = g.dead || (pl->m_player1 && pl->m_player1->m_isDead);
    g = Session{};
    if (dead) pl->resetLevel();
    log::info("[agent] trainer gone, level handed back to the player");
}

int gamemode(PlayerObject* p) {
    if (p->m_isShip)   return 1;
    if (p->m_isBall)   return 2;
    if (p->m_isBird)   return 3;  // UFO
    if (p->m_isDart)   return 4;  // wave
    if (p->m_isRobot)  return 5;
    if (p->m_isSpider) return 6;
    if (p->m_isSwing)  return 7;
    return 0;                     // cube
}

// Which object types matter for gameplay. Everything else (decoration,
// triggers, coins, ...) is dropped so the agent only "sees" what can kill it,
// hold it up, or change how it moves.
bool isGameplayType(GameObjectType t) {
    switch (t) {
        case GameObjectType::Solid:
        case GameObjectType::Breakable:
        case GameObjectType::Slope:
        case GameObjectType::Hazard:
        case GameObjectType::AnimatedHazard:
        case GameObjectType::YellowJumpPad:
        case GameObjectType::PinkJumpPad:
        case GameObjectType::GravityPad:
        case GameObjectType::RedJumpPad:
        case GameObjectType::SpiderPad:
        case GameObjectType::YellowJumpRing:
        case GameObjectType::PinkJumpRing:
        case GameObjectType::GravityRing:
        case GameObjectType::GreenRing:
        case GameObjectType::DropRing:
        case GameObjectType::RedJumpRing:
        case GameObjectType::CustomRing:
        case GameObjectType::DashRing:
        case GameObjectType::GravityDashRing:
        case GameObjectType::SpiderOrb:
        case GameObjectType::TeleportOrb:
        case GameObjectType::InverseGravityPortal:
        case GameObjectType::NormalGravityPortal:
        case GameObjectType::GravityTogglePortal:
        case GameObjectType::ShipPortal:
        case GameObjectType::CubePortal:
        case GameObjectType::BallPortal:
        case GameObjectType::UfoPortal:
        case GameObjectType::WavePortal:
        case GameObjectType::RobotPortal:
        case GameObjectType::SpiderPortal:
        case GameObjectType::SwingPortal:
        case GameObjectType::InverseMirrorPortal:
        case GameObjectType::NormalMirrorPortal:
        case GameObjectType::RegularSizePortal:
        case GameObjectType::MiniSizePortal:
        case GameObjectType::DualPortal:
        case GameObjectType::SoloPortal:
        case GameObjectType::TeleportPortal:
            return true;
        default:
            return false;
    }
}

bool collides(GameObjectType t) {
    return t == GameObjectType::Solid || t == GameObjectType::Breakable ||
           t == GameObjectType::Slope || t == GameObjectType::Hazard ||
           t == GameObjectType::AnimatedHazard;
}

// {"type":"level","length":L,"objects":[[type,x,y,w,h],...]}
// x,y is the bottom-left corner of the object's hitbox.
void sendLevel(PlayLayer* pl) {
    std::string msg;
    msg.reserve(64 * 1024);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "{\"type\":\"level\",\"length\":%.1f,\"objects\":[",
                  pl->m_levelLength);
    msg += buf;

    int kept = 0, total = 0;
    if (auto objs = pl->m_objects) {
        total = static_cast<int>(objs->count());
        for (unsigned i = 0; i < objs->count(); i++) {
            auto obj = static_cast<GameObject*>(objs->objectAtIndex(i));
            if (!obj || !isGameplayType(obj->m_objectType)) continue;
            // Solids/hazards flagged as non-colliding are just scenery.
            if (collides(obj->m_objectType) && (obj->m_isNoTouch || obj->m_isPassable)) continue;

            auto const& r = obj->getObjectRect();
            std::snprintf(buf, sizeof(buf), "%s[%d,%.1f,%.1f,%.1f,%.1f]",
                          kept ? "," : "", static_cast<int>(obj->m_objectType),
                          r.origin.x, r.origin.y, r.size.width, r.size.height);
            msg += buf;
            kept++;
        }
    }
    msg += "]}";
    bridge::send(msg);
    bridge::flush(LEVEL_FLUSH_MS);
    log::info("[agent] sent level: {} gameplay objects (of {}), length {:.0f}",
              kept, total, pl->m_levelLength);
}

void sendState(PlayLayer* pl) {
    auto p = pl->m_player1;
    if (!p) return;
    bool dead = g.dead || p->m_isDead;
    float pct = pl->getCurrentPercent();
    bool won = g.won || pct >= 100.f;
    char buf[320];
    std::snprintf(buf, sizeof(buf),
        "{\"type\":\"state\",\"x\":%.3f,\"y\":%.3f,\"vy\":%.4f,\"ground\":%d,"
        "\"mode\":%d,\"upside\":%d,\"held\":%d,\"speed\":%.3f,\"pct\":%.3f,"
        "\"dead\":%d,\"win\":%d}",
        p->getPositionX(), p->getPositionY(), static_cast<double>(p->m_yVelocity),
        p->m_isOnGround ? 1 : 0, gamemode(p), p->m_isUpsideDown ? 1 : 0,
        g.held ? 1 : 0, p->m_playerSpeed, pct, dead ? 1 : 0, won ? 1 : 0);
    bridge::send(buf);
}

// Returns the action in {"cmd":"step","a":N}, or -1 if the line isn't a step.
int parseStep(std::string const& line) {
    if (line.find("\"step\"") == std::string::npos) return -1;
    auto pos = line.find("\"a\":");
    if (pos == std::string::npos) return -1;
    pos += 4;
    while (pos < line.size() && line[pos] == ' ') pos++;
    if (pos >= line.size()) return -1;
    int a = line[pos] - '0';
    return (a >= NOTHING && a <= HOLD) ? a : -1;
}

} // namespace

class $modify(AgentBaseLayer, GJBaseGameLayer) {
    void update(float dt) {
        auto pl = PlayLayer::get();
        // Only drive real levels (not the editor), and only when enabled.
        if (!pl || static_cast<GJBaseGameLayer*>(pl) != this || !agentEnabled()) {
            GJBaseGameLayer::update(dt);
            return;
        }

        if (!bridge::connected()) {
            if (g.layer == pl) handBack(pl);
            if (g_frame++ % CONNECT_EVERY == 0 && bridge::tryConnect()) {
                g = Session{};
                g.layer = pl;
            }
            if (!bridge::connected()) {
                GJBaseGameLayer::update(dt);  // no trainer: play normally
                return;
            }
        }
        if (g.layer != pl) {  // a new level was opened on the same connection
            g = Session{};
            g.layer = pl;
        }

        serve(pl);
    }

    // Handle up to "steps-per-frame" commands within FRAME_BUDGET_MS. If no
    // command is waiting, the game simply doesn't advance this frame.
    void serve(PlayLayer* pl) {
        using clock = std::chrono::steady_clock;
        auto start = clock::now();
        int maxCommands = static_cast<int>(Mod::get()->getSettingValue<int64_t>("steps-per-frame"));
        int handled = 0;

        while (bridge::connected() && handled < maxCommands) {
            auto line = bridge::nextLine();
            if (!line) {
                int used = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                    clock::now() - start).count());
                int left = FRAME_BUDGET_MS - used;
                if (left <= 0 || !bridge::waitReadable(left)) break;
                continue;
            }

            if (line->find("\"reset\"") != std::string::npos) {
                doReset(pl);
            } else if (int a = parseStep(*line); a >= 0) {
                doStep(pl, a);
            } else {
                log::warn("[agent] unknown command: {}", *line);
                continue;
            }
            sendState(pl);
            handled++;
        }
        bridge::flush(0);
    }

    void doReset(PlayLayer* pl) {
        setButton(this, false);
        pl->resetLevel();
        g.dead = false;
        g.won = false;
        if (!g.levelSent) {
            sendLevel(pl);
            g.levelSent = true;
        }
    }

    void doStep(PlayLayer* pl, int action) {
        if (g.dead || g.won) return;  // episode is over; wait for reset
        for (int i = 0; i < FRAME_SKIP; i++) {
            bool down = (action == HOLD) || (action == CLICK && i == 0);
            // A click while the button is already held needs a release first,
            // otherwise the game never sees a fresh press.
            if (action == CLICK && i == 0 && g.held) setButton(this, false);
            setButton(this, down);
            GJBaseGameLayer::update(FIXED_DT);
            if (g.dead || g.won || (pl->m_player1 && pl->m_player1->m_isDead)) break;
        }
    }
};

class $modify(AgentPlayLayer, PlayLayer) {
    void destroyPlayer(PlayerObject* player, GameObject* obj) {
        // GD calls destroyPlayer(player, m_anticheatSpike) at the start of
        // every attempt as a noclip check. That is not a death.
        bool real = obj != m_anticheatSpike;
        PlayLayer::destroyPlayer(player, obj);
        if (real && agentActive()) g.dead = true;
    }

    void delayedResetLevel() {
        // The trainer decides when to restart (it sends "reset").
        if (agentActive()) return;
        PlayLayer::delayedResetLevel();
    }

    void levelComplete() {
        // Flag the win and skip the end-of-level sequence so the next reset
        // starts a clean attempt.
        if (agentActive()) {
            g.won = true;
            return;
        }
        PlayLayer::levelComplete();
    }

    void onQuit() {
        // Leaving the level closes the socket; the trainer sees EOF and waits
        // for the next level to connect.
        if (bridge::connected()) bridge::disconnect();
        g = Session{};
        PlayLayer::onQuit();
    }

    void pauseGame(bool unfocused) {
        // GD pauses the level when its window loses focus. While the agent is
        // training, ignore that so you can watch from other windows.
        // Esc still pauses (that path passes unfocused = false).
        if (unfocused && agentActive()) return;
        PlayLayer::pauseGame(unfocused);
    }
};

// Focus loss reaches the level through the app delegate first
// (applicationWillResignActive -> AppDelegate::pauseGame -> PlayLayer::pauseGame(true)).
// Block it at each step, but only while the agent is in control.
class $modify(AgentAppDelegate, AppDelegate) {
    void applicationWillResignActive() {
        if (agentActive()) return;
        AppDelegate::applicationWillResignActive();
    }

    void pauseGame() {
        if (agentActive()) return;
        AppDelegate::pauseGame();
    }
};

$on_mod(Loaded) {
    log::info("GD AI 2 Bridge loaded (socket {})", bridge::SOCK_PATH);
}
