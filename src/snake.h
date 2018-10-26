#pragma once

#include <vector>
#include "json.hpp"
#include "mlp.hpp"

#define WIDTH 46
#define HEIGHT 34

enum ECell {
  CELL_EMPTY = 0,
  CELL_FOOD,
  CELL_OBSTACLE,
  CELL_HEAD,
  CELL_BODY,
  CELL_TAIL
};

enum EInput {
  I_EMPTY = 0,
  I_MAX_DIST,
  I_MIN_DIST,
  I_FOOD,
  I_OBSTACLE,
  I_HEAD,
  I_BODY,
  I_TAIL,
  I_DAZME,
  NUM_INPUTS,
};

enum EDirection {
  DIRECTION_UP = 0,
  DIRECTION_DOWN,
  DIRECTION_LEFT,
  DIRECTION_RIGHT,
};

// Determine how far the snake sees these objects
#define SENSE_EMPTY 0.5
#define SENSE_MAX_DIST 1
#define SENSE_MIN_DIST 1
#define SENSE_FOOD 0.5
#define SENSE_OBSTACLE 1
#define SENSE_HEAD 1
#define SENSE_BODY 0.5
#define SENSE_TAIL 1
#define SENSE_DAZME 0.5

struct Cell {
  Cell *edges[4] = { NULL, NULL, NULL, NULL };

  Cell *prev = NULL;
  unsigned dist = 0;

  ECell type;
  bool dazme = false;
  bool is_endpoint = false;
  
  long double inputs[NUM_INPUTS];
  
  void clear_tile() {
    type = CELL_EMPTY;
    dazme = false;
    is_endpoint = false;
  };

  void clear_sums() {
    prev = NULL;
    dist = 1;
    for (unsigned i=0; i<NUM_INPUTS; ++i) {
      inputs[i] = 0.0;
    }
  }
  
  void set_food() {
    type = CELL_FOOD;
  }

  void set_obstacle() {
    type = CELL_OBSTACLE;
    is_endpoint = true;
  }

  void set_body(bool _dazme) {
    type = CELL_BODY;
    dazme = _dazme;
    is_endpoint = true;
  }

  void set_head(bool _dazme) {
    type = CELL_HEAD;
    dazme = _dazme;
    is_endpoint = true;
  }

  void set_tail(long double _dazme) {
    type = CELL_TAIL;
    dazme = _dazme;
    is_endpoint = true;
  }

  void sum_up(Cell *cell) {
    long double fade = 1.0 / cell->dist;

    switch (cell->type) {
    case CELL_EMPTY:    inputs[I_EMPTY]    += SENSE_EMPTY    * fade; break;
    case CELL_FOOD:     inputs[I_FOOD]     += SENSE_FOOD     * fade; break;
    case CELL_OBSTACLE: inputs[I_OBSTACLE] += SENSE_OBSTACLE * fade; break;
    case CELL_HEAD:     inputs[I_HEAD]     += SENSE_HEAD     * fade; break;
    case CELL_BODY:     inputs[I_BODY]     += SENSE_BODY     * fade; break;
    case CELL_TAIL:     inputs[I_TAIL]     += SENSE_TAIL     * fade; break;
    }

    if (cell->dazme)    inputs[I_DAZME]    += SENSE_DAZME    * fade;

    if (cell->dist > inputs[I_MAX_DIST]) {
      inputs[I_MAX_DIST] = cell->dist;
    }
  }
};

class Snake
{
private:
  Cell cells[HEIGHT*WIDTH];
  MLP<long double> mlp;

  int uid;
  std::string player_id;

  unsigned board_cells;
  Cell *my_head;

  void set_edges_(Cell *cell, int x, int y);
  void make_cells_();
  void load_map_(nlohmann::json map);
  void compute_sums_(Cell *start);
  long double get_q_value_(Cell *cell);

public:
  int age;
  int points;
  bool is_alive;
  
  Snake();

  void set_parameters(std::vector<long double> &x);
  
  std::string name = "Neurotic";
  std::string get_next_move(nlohmann::json &map);
  std::string game_link;
  void on_game_ended();
  void on_tournament_ended();
  void on_snake_dead(std::string death_reason);
  void on_game_starting();
  void on_player_registered(std::string &_player_id);
  void on_game_link(std::string &_game_link);
  void on_invalid_playername();
  void on_game_result(nlohmann::json &playerRanks);
};
