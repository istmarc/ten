#ifndef TEN_DATETIME_HXX
#define TEN_DATETIME_HXX

#include <iostream>

#include <ten/types.hxx>

namespace ten {

/// Month of the year
enum class month : unsigned char {
  jan = 0,
  feb = 1,
  mar = 2,
  apr = 3,
  may = 4,
  jun = 5,
  jul = 6,
  aug = 7,
  sep = 8,
  oct = 9,
  nov = 10,
  dec = 11
};

// Overload the << operator for month
std::ostream &operator<<(std::ostream &os, const month m) {
  switch (m) {
  case month::jan:
    os << "jan";
    break;
  case month::feb:
    os << "feb";
    break;
  case month::mar:
    os << "mar";
    break;
  case month::apr:
    os << "apr";
    break;
  case month::may:
    os << "may";
    break;
  case month::jun:
    os << "jun";
    break;
  case month::jul:
    os << "jul";
    break;
  case month::aug:
    os << "aug";
    break;
  case month::sep:
    os << "sep";
    break;
  case month::oct:
    os << "oct";
    break;
  case month::nov:
    os << "nov";
    break;
  case month::dec:
    os << "dec";
    break;
  }
  return os;
}

/// Day of the week
enum class day : unsigned char {
  mon = 0,
  tue = 1,
  wed = 2,
  thu = 3,
  fri = 4,
  sat = 5,
  sun = 6
};

// Overload the << operator for day
std::ostream &operator<<(std::ostream &os, const day d) {
  switch (d) {
  case day::mon:
    os << "mon";
    break;
  case day::tue:
    os << "tue";
    break;
  case day::wed:
    os << "wed";
    break;
  case day::thu:
    os << "thu";
    break;
  case day::fri:
    os << "fri";
    break;
  case day::sat:
    os << "sat";
    break;
  case day::sun:
    os << "sun";
    break;
  }
  return os;
}

/// Date
struct date {
  long year;
  ten::month month;
  std::size_t day;

  date(long y, ten::month m, std::size_t d) : year(y), month(m), day(d) {}
};

/// Overload the << operator for date
std::ostream &operator<<(std::ostream &os, const date &d) {
  os << d.year;
  os << "-";
  os << d.month;
  os << "-";
  os << d.day;
  return os;
}

} // namespace ten

#endif
