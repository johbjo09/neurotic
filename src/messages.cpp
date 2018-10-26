#include "messages.h"
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/utsname.h>

using nlohmann::json;

json start_game() {
  json start_game_msg;
  start_game_msg["type"] = START_GAME;

  return start_game_msg;
}

std::tuple<std::string, std::string> get_os_info() {
#if defined(_WIN32) || defined(_WIN64)
  OSVERSIONINFO osvi;
  BOOL bIsWindowsXPorLater;

  ZeroMemory(&osvi, sizeof(OSVERSIONINFO));
  osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);

  GetVersionEx(&osvi);

  return std::make_tuple("Windows", osvi.dwMajorVersion + "." + osvi.dwMinorVersion);
#elif __linux__ || __FreeBSD__
  struct utsname osinfo;
  uname(&osinfo);
  return std::make_tuple(osinfo.sysname, osinfo.version);
#elif __APPLE__ || __MACH__
  // Yep, that appears to be the completely obvious way to do it
  std::string version = "version";
  return std::make_tuple("Mac OS X", version);
#else
  return std::make_tuple("", "");
#endif
}

json client_info() {
  std::string os_name, os_version;
  std::tie(os_name, os_version) = get_os_info();
  json client_info_msg;
  client_info_msg["type"] = CLIENT_INFO;
  client_info_msg["language"] = "C++";
  client_info_msg["languageVersion"] = "11";
  client_info_msg["operatingSystem"] = os_name;
  client_info_msg["operatingSystemVersion"] = os_version;
  client_info_msg["clientVersion"] = "1.0";

  return client_info_msg;
}

json default_game_settings() {
  json game_settings;
  game_settings["maxNoofPlayers"] = 5;
  game_settings["startSnakeLenth"] = 1;
  game_settings["timeInMsPerTick"] = 250;
  game_settings["obstaclesEnabled"] = true;
  game_settings["foodEnabled"] = true;
  game_settings["headToTailConsumes"] = true;
  game_settings["tailConsumeGrows"] = false;
  game_settings["addFoodLikelihood"] = 15;
  game_settings["removeFoodLikelihood"] = 5;
  game_settings["spontaneousGrowthEveryNWorldTick"] = 3;
  game_settings["trainingGame"] = false;
  game_settings["pointsPerLength"] = 1;
  game_settings["pointsPerFood"] = 2;
  game_settings["pointsPerCausedDeath"] = 5;
  game_settings["pointsPerNibble"] = 10;
  game_settings["noofRoundsTailProtectedAfterNibble"] = 3;

  return game_settings;
}

json player_registration(std::string snake_name) {
  json player_registration_msg;
  player_registration_msg["type"] = REGISTER_PLAYER_MESSAGE_TYPE;
  player_registration_msg["playerName"] = snake_name;
  player_registration_msg["gameSettings"] = default_game_settings();

  return player_registration_msg;
}

json register_move(std::string next_move, json incoming_json) {
  json register_move_msg;
  register_move_msg["type"] = REGISTER_MOVE;
  register_move_msg["direction"] = next_move;
  register_move_msg["gameTick"] = incoming_json["gameTick"];
  register_move_msg["receivingPlayerId"] = incoming_json["receivingPlayerId"];
  register_move_msg["gameId"] = incoming_json["gameId"];

  return register_move_msg;
}

json heart_beat(std::string id) {
  json heart_beat;
  heart_beat["type"] = HEART_BEAT_REQUEST;
  heart_beat["receivingPlayerId"] = id;

  return heart_beat;
}
