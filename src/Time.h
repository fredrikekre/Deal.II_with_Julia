//
// Created by kristoffer on 2015-12-01.
//

#ifndef PROJECT_TIME_H
#define PROJECT_TIME_H

class Time {
public:
  Time(const double time_endd, const double delta_tt)
      : timestep(0), time_current(0.0), time_end(time_endd), delta_t(delta_tt) {
  }
  virtual ~Time() {}
  double current() const { return time_current; }
  double end() const { return time_end; }
  /* double get_delta_t() const { return delta_t; } */
  unsigned int get_timestep() const { return timestep; }
  void increment() {
    time_current += delta_t;
    ++timestep;
  }
  void reset() { time_current = 0.0; }

private:
  unsigned int timestep;
  double time_current;
  const double time_end;
  const double delta_t;
};

#endif // PROJECT_TIME_H
