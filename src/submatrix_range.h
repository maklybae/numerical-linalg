#ifndef SUBMATRIX_RANGE_H
#define SUBMATRIX_RANGE_H

#include <cassert>

#include "types_details.h"

namespace linalg::types {

class SubmatrixRange {
 public:
  SubmatrixRange() = default;

  Size RowBegin() const {
    return row_begin_;
  }

  Size RowEnd() const {
    return row_end_;
  }

  Size ColBegin() const {
    return col_begin_;
  }

  Size ColEnd() const {
    return col_end_;
  }

  Size Rows() const {
    assert(row_begin_ < row_end_ && "Row begin should be less than row end");

    return row_end_ - row_begin_;
  }

  Size Cols() const {
    assert(col_begin_ < col_end_ && "Col begin should be less than col end");

    return col_end_ - col_begin_;
  }

  static SubmatrixRange LeadingSubmatrix(Size rows, Size cols) {
    return SubmatrixRange{0, rows, 0, cols};
  }

  static SubmatrixRange LeadingSubmatrix(Size size) {
    return LeadingSubmatrix(size, size);
  }

  static SubmatrixRange FullMatrix(Size rows, Size cols) {
    return SubmatrixRange{0, rows, 0, cols};
  }

  static SubmatrixRange FromBeginEnd(Index row_begin, Index row_end, Index col_begin, Index col_end) {
    return SubmatrixRange{row_begin, row_end, col_begin, col_end};
  }

  static SubmatrixRange FromBeginSize(Index row_begin, Size rows, Index col_begin, Size cols) {
    return SubmatrixRange{row_begin, row_begin + rows, col_begin, col_begin + cols};
  }

 private:
  SubmatrixRange(Size row_begin, Size row_end, Size col_begin, Size col_end)
      : row_begin_{row_begin}, row_end_{row_end}, col_begin_{col_begin}, col_end_{col_end} {
    assert(row_begin < row_end && "Row begin should be less than row end");
    assert(col_begin < col_end && "Col begin should be less than col end");
  }

  Index row_begin_{};
  Index row_end_{};
  Index col_begin_{};
  Index col_end_{};
};

}  // namespace linalg::types

#endif
