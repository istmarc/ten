#ifndef TEN_DATETIME_HXX
#define TEN_DATETIME_HXX

#include <ten/types.hxx>

namespace ten{

/// Month of the year
enum class month : unsigned char{
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

/// Day of the week
enum class day : unsigned char{
  mon = 0,
  tue = 1,
  wed = 2,
  thu = 3,
  fri = 4,
  sat = 5,
  sun = 6
};

/// Date
struct date{
  long year;
  ten::month month;
  std::size_t day;

  date(long y, ten::month m, std::size_t d) : year(y), month(m), day(d) {}

};

}

#endif
