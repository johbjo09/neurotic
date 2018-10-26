#include <queue>
#include "easylogging++.h"
#include "json.hpp"
#include "snake.h"

using nlohmann::json;

static const long double mean_inputs[] = { (HEIGHT * WIDTH) / 10.0,
					   15,
					   10,
					   1,
					   1,
					   1,
					   2,
					   1.5,
					   2 };

static const std::string DIRECTION_STR[4] = {
  "UP",
  "DOWN",
  "LEFT",
  "RIGHT",
};

Snake::Snake()
  : mlp(NUM_INPUTS, ACTIVATION_TANH),
    board_cells(WIDTH*HEIGHT)
{
  
  for (int y=0; y<HEIGHT; ++y) {
    for (int x=0; x<WIDTH; ++x) {
      Cell *cell = &cells[x + y*WIDTH];
      set_edges_(cell, x, y);
    }
  }

  // Deep snake
  mlp.add_layer(16);
  mlp.add_layer(7);
  mlp.add_layer(1);
}

void Snake::set_parameters(std::vector<long double> &parameters)
{
  mlp.set_parameters(parameters);
}

void Snake::set_edges_(Cell *cell, int x, int y)
{
  // Up
  if (y > 0) {
    cell->edges[DIRECTION_UP] = &cells[x + (y-1) * WIDTH];
  }

  // Down
  if (y < HEIGHT-1) {
    cell->edges[DIRECTION_DOWN] = &cells[x + (y+1) * WIDTH];
  }

  // Left
  if (x > 1) {
    cell->edges[DIRECTION_LEFT] = &cells[x-1 + y*WIDTH];
  }

  // Right
  if (x < WIDTH-1) {
    cell->edges[DIRECTION_RIGHT] = &cells[x+1 + y*WIDTH];
  }
}

static const char CELL_CHAR[6] = { '_', '*', '#', 'H', 'B', 'T' };
static const char CELL_CHAR_ME[6] = { '_', '*', '#', 'h', 'b', 't' };

void print_board(Cell *cells)
{
  for (int y=0; y<HEIGHT; ++y) {
    for (int x=0; x<WIDTH; ++x) {
      Cell *cell = &cells[x + y*WIDTH];
      if (cell->dazme) {
	std::cerr << CELL_CHAR_ME[cell->type] << (cell->is_endpoint ? "." : " ");
      } else {
	std::cerr << CELL_CHAR[cell->type] << (cell->is_endpoint ? "." : " ");
      }
    }
    std::cerr << "\n";
  }
}

void Snake::load_map_(nlohmann::json map)
{
  for (unsigned i=0; i<board_cells; ++i) {
    cells[i].clear_tile();
  }

  for (auto &snake: map["snakeInfos"]) {
    auto &positions = snake["positions"];

    bool dazme = false;


    Cell *head = &cells[(int) positions.front()];
    if (player_id.compare(snake["id"]) == 0) {
      dazme = true;
      my_head = head;
    }

    for (auto &json_position: positions) {
      int position = (int) json_position;
      cells[position].set_body(dazme);
    }
    
    head->set_head(dazme);
    cells[(int) positions.back()].set_tail(dazme);

    // Not ok. Should not include head/tail in body
  }

  for (auto &json_position: map["obstaclePositions"]) {
    int position = (int) json_position;
    cells[position].set_obstacle();
  }

  for (auto &json_position: map["foodPositions"]) {
    int position = (int) json_position;
    cells[position].set_food();
  }
}

void Snake::compute_sums_(Cell *start)
{
  for (unsigned i = 0; i<board_cells; ++i) {
    cells[i].clear_sums();
  }

  start->prev = start;
  std::queue<Cell*> frontier;

  frontier.push(start);

  while (!frontier.empty()) {
    Cell *cell = frontier.front();
    frontier.pop();
    start->sum_up(cell);

    for (int p=0; p<4; ++p) {
      Cell *vertex = cell->edges[p];
      if (vertex == NULL)
	continue;
      if (vertex->prev == NULL) {
	vertex->prev = cell;
	vertex->dist = cell->dist + 1;
	if (!vertex->is_endpoint) {
	  frontier.push(vertex);
	}
      }
    }
  }
}

long double Snake::get_q_value_(Cell *cell)
{
  compute_sums_(cell);

  std::vector<long double> inputs (NUM_INPUTS, 0.0);

  for (unsigned i=0; i<NUM_INPUTS; i++) {
    inputs[i] = cell->inputs[i] / mean_inputs[i];
  }
  
  std::vector<long double> outputs (1, 0);

  mlp.recall(inputs, outputs);

  long double q_value = outputs[0];

  return q_value;
}

std::string Snake::get_next_move(json &map)
{
  age++;
  load_map_(map);

  long double q[4];
  
  for (unsigned i=0; i<4; ++i) {
    q[i] = -1e10;
    Cell *cell_dir = my_head->edges[i];
    if (cell_dir != NULL && !cell_dir->is_endpoint) {
      q[i] = get_q_value_(cell_dir);
    }
  }

  int best_dir = 0;
  long double max_q = -1e10;
  for (unsigned i=0; i<4; ++i) {
    if (q[i] > max_q) {
      best_dir = i;
      max_q = q[i];
    }
  }

  std::string response = DIRECTION_STR[best_dir];

  LOG(INFO) << "Snake is making move " << response << " at worldtick: " << map["worldTick"];
  
  return response;
}

void Snake::on_game_ended() {
  LOG(INFO) << "Game has ended";
};

void Snake::on_tournament_ended() {
  LOG(INFO) << "Tournament has ended";
};

void Snake::on_snake_dead(std::string death_reason) {
  LOG(INFO) << "Our snake has died, reason was: " << death_reason;
};

void Snake::on_game_starting() {
  LOG(INFO) << "Game is starting";
  age = 0;
  points = 0;
  is_alive = false;
};

void Snake::on_player_registered(std::string &_player_id) {
  player_id = _player_id;
  LOG(INFO) << "Player was successfully registered";
};

void Snake::on_game_link(std::string &_game_link) {
  game_link = _game_link;
  LOG(INFO) << "Player was successfully registered";
};

void Snake::on_invalid_playername() {
  LOG(INFO) << "The player name is invalid, try another?";
};

void Snake::on_game_result(nlohmann::json &player_ranks) {
  LOG(INFO) << "Game result:";
  nlohmann::json playerRank;

  /*
  el::Logger* defaultLogger = el::Loggers::getLogger("default");
  for (json::iterator it = playerRanks.begin(); it != playerRanks.end(); ++it) {
    playerRank = (nlohmann::json) *it;
    defaultLogger->info("%v.\t%v pts\t%v (%v)", playerRank["rank"], playerRank["points"],
            playerRank["playerName"], playerRank["alive"] ? "alive" : "dead");
  }
  */
  
  for (auto &player: player_ranks) {
    if (name.compare(player["playerName"]) == 0) {
      is_alive = player["alive"];
      points = player["points"];
    }
  }
};

