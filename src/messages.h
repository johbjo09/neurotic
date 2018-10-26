#pragma once

#include "json.hpp"

// Inbound
static const std::string GAME_ENDED = "se.cygni.snake.api.event.GameEndedEvent";
static const std::string TOURNAMENT_ENDED = "se.cygni.snake.api.event.TournamentEndedEvent";
static const std::string MAP_UPDATE = "se.cygni.snake.api.event.MapUpdateEvent";
static const std::string SNAKE_DEAD = "se.cygni.snake.api.event.SnakeDeadEvent";
static const std::string GAME_STARTING = "se.cygni.snake.api.event.GameStartingEvent";
static const std::string PLAYER_REGISTERED = "se.cygni.snake.api.response.PlayerRegistered";
static const std::string INVALID_PLAYER_NAME = "se.cygni.snake.api.exception.InvalidPlayerName";
static const std::string HEART_BEAT_RESPONSE = "se.cygni.snake.api.response.HeartBeatResponse";
static const std::string GAME_LINK_EVENT = "se.cygni.snake.api.event.GameLinkEvent";
static const std::string GAME_RESULT_EVENT = "se.cygni.snake.api.event.GameResultEvent";

// Outbound
static const std::string REGISTER_PLAYER_MESSAGE_TYPE = "se.cygni.snake.api.request.RegisterPlayer";
static const std::string START_GAME = "se.cygni.snake.api.request.StartGame";
static const std::string REGISTER_MOVE = "se.cygni.snake.api.request.RegisterMove";
static const std::string HEART_BEAT_REQUEST = "se.cygni.snake.api.request.HeartBeatRequest";
static const std::string CLIENT_INFO = "se.cygni.snake.api.request.ClientInfo";

nlohmann::json start_game();
nlohmann::json client_info();
nlohmann::json default_game_settings();
nlohmann::json player_registration(std::string snake_name);
nlohmann::json register_move(std::string next_move, nlohmann::json incoming_json);
nlohmann::json heart_beat(std::string id);
