#pragma once

/**
Customisable progressbar decorator for iterators.
Includes a default range iterator printing to stderr.

TODO:
* handle s(tep) with operator+/-(=)
* chrono and delay printing
* class status printer
* iterator-less: fake range iterable, update() increments value

Usage:
  # include "tqdm/tqdm.h"
  for(int i : tqdm::range(4))
  // same as:
  //   for (int i : tqdm::tqdm({0, 1, 2, 3}))
  // or:
  //   std::vector<int> v{0, 1, 2, 3};
  //   for (int &i : tqdm::tqdm(v.begin(), v.end())
    ...

@author Casper dC-L <github.com/casperdcl>
*/

#include <cassert>      // assert
#include <cinttypes>    // PRIu64
#include <cstddef>      // ptrdiff_t, size_t
#include <cstdint>      // int64_t
#include <cstdio>       // printf
#include <iterator>     // iterator
#include <limits>       // numeric_limits
#include <stdexcept>    // throw
#include <string>       // string
#include <type_traits>  // is_pointer, ...
#include <utility>      // swap
#include "tqdm_utils.h"

#ifndef SIZE_T_MAX
constexpr size_t SIZE_T_MAX = std::numeric_limits<size_t>::max();
#endif

namespace tqdm {

struct Params {
  std::string desc;
  size_t total = -1;
  bool leave = true;
  FILE* f = stderr;
  int ncols = -1;
  float mininterval = 0.1f, maxinterval = 10.0f;
  unsigned miniters = -1;
  std::string ascii = " 123456789#";
  bool disable = false;
  std::string unit = "it";
  bool unit_scale = false;
  bool dynamic_ncols = false;
  float smoothing = 0.3f;
  std::string bar_format;
  size_t initial = 0;
  int position = -1;
  bool gui = false;
};

template <typename _Iterator>
class Tqdm : public MyIteratorWrapper<_Iterator> {
 private:
  using TQDM_IT = MyIteratorWrapper<_Iterator>;
  _Iterator e;  // end
  Params self;  // ha, ha

 public:
  /**
   containter-like methods
   */
  // actually current value
  // virtual _Iterator begin() { return this->get(); }
  Tqdm& begin() { return *this; }
  const Tqdm& begin() const { return *this; }
  // virtual _Iterator end() { return e; }
  Tqdm end() const { return Tqdm(e, e); }

  explicit operator _Iterator() { return this->get(); }

  /** constructors
   */
  explicit Tqdm(_Iterator begin, _Iterator end)
      : TQDM_IT(begin), e(end), self() {
    self.total = size_t(end - begin);
  }

  explicit Tqdm(_Iterator begin, size_t total)
      : TQDM_IT(begin), e(begin + total), self() {
    self.total = total;
  }

  // Tqdm(const Tqdm& other)
  //     : TQDM_IT(other.get()),
  //       e(other.end().get()),
  //       self(other.self),
  // {
  //   // std::memcpy(this, &other, sizeof(Tqdm));
  // }

  template <typename _Container,
            typename = typename std::enable_if<
                !std::is_same<_Container, Tqdm>::value>::type>
  Tqdm(_Container& v) : TQDM_IT(std::begin(v)), e(std::end(v)), self() {
    self.total = e - this->get();
  }

  explicit operator bool() const { return this->get() != e; }

  /** TODO: magic methods */
  virtual void _incr() const override {
    if (this->get() == e)
      throw std::out_of_range(
          "exhausted");  // TODO: don't throw, just double total

    TQDM_IT::_incr();
    if (this->get() == e) {
      printf("\nfinished: %" PRIu64 "/%" PRIu64 "\n",
             static_cast<std::uint64_t>(self.total),
             static_cast<std::uint64_t>(self.total));
    } else
      printf("\r%" PRIi64 " left", (int64_t)(e - this->get()));
  }
  virtual void _incr() override { ((Tqdm const&)*this)._incr(); }
};

template <typename _Iterator, typename _Tqdm = Tqdm<_Iterator>>
_Tqdm tqdm(_Iterator begin, _Iterator end) {
  return _Tqdm(begin, end);
}

template <typename _Iterator, typename _Tqdm = Tqdm<_Iterator>>
_Tqdm tqdm(_Iterator begin, size_t total) {
  return _Tqdm(begin, total);
}

template <typename _Container,
          typename _Tqdm = Tqdm<typename _Container::iterator>>
_Tqdm tqdm(_Container& v) {
  return _Tqdm(v);
}

template <size_t N, typename T, typename _Tqdm = Tqdm<T*>>
_Tqdm tqdm(T (&tab)[N]) {
  return _Tqdm(tab, N);
}

template <typename SizeType = int>
using RangeTqdm = Tqdm<RangeIterator<SizeType>>;
template <typename SizeType>
RangeTqdm<SizeType> range(SizeType n) {
  return RangeTqdm<SizeType>(RangeIterator<SizeType>(n),
                             RangeIterator<SizeType>(n));
}
template <typename SizeType>
RangeTqdm<SizeType> range(SizeType start, SizeType end) {
  return RangeTqdm<SizeType>(RangeIterator<SizeType>(start, end),
                             RangeIterator<SizeType>(start, end));
}
template <typename SizeType>
RangeTqdm<SizeType> range(SizeType start, SizeType end, SizeType step) {
  return RangeTqdm<SizeType>(RangeIterator<SizeType>(start, end, step),
                             RangeIterator<SizeType>(start, end, step));
}

}  // namespace tqdm

/** Things to port:

class TqdmTypeError(TypeError):
class TqdmKeyError(KeyError):
class TqdmDeprecationWarning(Exception):
ASCII_FMT = " 123456789#"
UTF_FMT = u" " + u''.join(map(_unich, range(0x258F, 0x2587, -1)))
class tqdm(object):
    @staticmethod
    def format_sizeof(num, suffix=''):
    @staticmethod
    def format_interval(t):
    @staticmethod
    def status_printer(file):
    @staticmethod
    def format_meter(n, total, elapsed, ncols=None, prefix='',
                     ascii=False, unit='it', unit_scale=False, rate=None,
                     bar_format=None):
    def __new__(cls, *args, **kwargs):
    @classmethod
    def _get_free_pos(cls, instance=None):
    @classmethod
    def _decr_instances(cls, instance):
    @classmethod
    def write(cls, s, file=sys.stdout, end="\n"):
    @classmethod
    def pandas(tclass, *targs, **tkwargs):
    def __init__(self, iterable=None, desc=None, total=None, leave=True,
                 file=sys.stderr, ncols=None, mininterval=0.1,
                 maxinterval=10.0, miniters=None, ascii=None, disable=False,
                 unit='it', unit_scale=False, dynamic_ncols=False,
                 smoothing=0.3, bar_format=None, initial=0, position=None,
                 gui=False, **kwargs):
        """
        Parameters
        ----------
        iterable  : iterable, optional
            Iterable to decorate with a progressbar.
            Leave blank to manually manage the updates.
        desc  : str, optional
            Prefix for the progressbar.
        total  : int, optional
            The number of expected iterations. If unspecified,
            len(iterable) is used if possible. As a last resort, only basic
            progress statistics are displayed (no ETA, no progressbar).
            If `gui` is True and this parameter needs subsequent updating,
            specify an initial arbitrary large positive integer,
            e.g. int(9e9).
        leave  : bool, optional
            If [default: True], keeps all traces of the progressbar
            upon termination of iteration.
        file  : `io.TextIOWrapper` or `io.StringIO`, optional
            Specifies where to output the progress messages
            [default: sys.stderr]. Uses `file.write(str)` and `file.flush()`
            methods.
        ncols  : int, optional
            The width of the entire output message. If specified,
            dynamically resizes the progressbar to stay within this bound.
            If unspecified, attempts to use environment width. The
            fallback is a meter width of 10 and no limit for the counter and
            statistics. If 0, will not print any meter (only stats).
        mininterval  : float, optional
            Minimum progress update interval, in seconds [default: 0.1].
        maxinterval  : float, optional
            Maximum progress update interval, in seconds [default: 10.0].
        miniters  : int, optional
            Minimum progress update interval, in iterations.
            If specified, will set `mininterval` to 0.
        ascii  : bool or str, optional
            If unspecified or False, use unicode (smooth blocks) to fill
            the meter.
            The fallback is to use ASCII characters " 123456789#".
        disable  : bool, optional
            Whether to disable the entire progressbar wrapper
            [default: False].
        unit  : str, optional
            String that will be used to define the unit of each iteration
            [default: it].
        unit_scale  : bool, optional
            If set, the number of iterations will be reduced/scaled
            automatically and a metric prefix following the
            International System of Units standard will be added
            (kilo, mega, etc.) [default: False].
        dynamic_ncols  : bool, optional
            If set, constantly alters `ncols` to the environment (allowing
            for window resizes) [default: False].
        smoothing  : float, optional
            Exponential moving average smoothing factor for speed estimates
            (ignored in GUI mode). Ranges from 0 (average speed) to 1
            (current/instantaneous speed) [default: 0.3].
        bar_format  : str, optional
            Specify a custom bar string formatting. May impact performance.
            If unspecified, will use '{l_bar}{bar}{r_bar}', where l_bar is
            '{desc}{percentage:3.0f}%|' and r_bar is
            '| {n_fmt}/{total_fmt} [{elapsed_str}<{remaining_str},
{rate_fmt}]'
            Possible vars: bar, n, n_fmt, total, total_fmt, percentage,
            rate, rate_fmt, elapsed, remaining, l_bar, r_bar, desc.
        initial  : int, optional
            The initial counter value. Useful when restarting a progress
            bar [default: 0].
        position  : int, optional
            Specify the line offset to print this bar (starting from 0)
            Automatic if unspecified.
            Useful to manage multiple bars at once (eg, from threads).
        gui  : bool, optional
            WARNING: internal parameter - do not use.
            Use tqdm_gui(...) instead. If set, will attempt to use
            matplotlib animations for a graphical output [default: False].

        Returns
        -------
        out  : decorated iterator.
        """
        if disable:
            self.iterable = iterable
            self.disable = disable
            self.pos = self._get_free_pos(self)
            self._instances.remove(self)
            return

        if kwargs:
            self.disable = True
            self.pos = self._get_free_pos(self)
            self._instances.remove(self)
            raise (TqdmDeprecationWarning("""\
`nested` is deprecated and automated. Use position instead for manual control.
""", fp_write=getattr(file, 'write', sys.stderr.write))
                if "nested" in kwargs else
                TqdmKeyError("Unknown argument(s): " + str(kwargs)))

        # Preprocess the arguments
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                total = None

        if ((ncols is None) and (file in (sys.stderr, sys.stdout))) or \
                dynamic_ncols:  # pragma: no cover
            if dynamic_ncols:
                dynamic_ncols = _environ_cols_wrapper()
                ncols = dynamic_ncols(file)
            else:
                ncols = _environ_cols_wrapper()(file)

        if miniters is None:
            miniters = 0
            dynamic_miniters = True
        else:
            dynamic_miniters = False

        if mininterval is None:
            mininterval = 0

        if maxinterval is None:
            maxinterval = 0

        if ascii is None:
            ascii = not _supports_unicode(file)

        if bar_format and not ((ascii is True) or _is_ascii(ascii)):
            # Convert bar format into unicode since terminal uses unicode
            bar_format = _unicode(bar_format)

        if smoothing is None:
            smoothing = 0

        # Store the arguments
        self.iterable = iterable
        self.desc = desc + ': ' if desc else ''
        self.total = total
        self.leave = leave
        self.fp = file
        self.ncols = ncols
        self.mininterval = mininterval
        self.maxinterval = maxinterval
        self.miniters = miniters
        self.dynamic_miniters = dynamic_miniters
        self.ascii = ascii
        self.disable = disable
        self.unit = unit
        self.unit_scale = unit_scale
        self.gui = gui
        self.dynamic_ncols = dynamic_ncols
        self.smoothing = smoothing
        self.avg_time = None
        self._time = time
        self.bar_format = bar_format

        # Init the iterations counters
        self.last_print_n = initial
        self.n = initial

        # if nested, at initial sp() call we replace '\r' by '\n' to
        # not overwrite the outer progress bar
        self.pos = self._get_free_pos(self) if position is None else position

        if not gui:
            # Initialize the screen printer
            self.sp = self.status_printer(self.fp)
            if self.pos:
                self.moveto(self.pos)
            self.sp(self.format_meter(self.n, total, 0,
                    (dynamic_ncols(file) if dynamic_ncols else ncols),
                    self.desc, ascii, unit, unit_scale, None, bar_format))
            if self.pos:
                self.moveto(-self.pos)

        # Init the time counter
        self.start_t = self.last_print_t = self._time()

    def __len__(self):
    def __enter__(self):
    def __exit__(self, *exc):
    def __del__(self):
    def __repr__(self):
    def __lt__(self, other):
    def __le__(self, other):
    def __eq__(self, other):
    def __ne__(self, other):
    def __gt__(self, other):
    def __ge__(self, other):
    def __hash__(self):
    def __iter__(self):
    def update(self, n=1):
        """
        Manually update the progress bar, useful for streams
        such as reading files.
        E.g.:
        >>> t = tqdm(total=filesize) # Initialise
        >>> for current_buffer in stream:
        ...    ...
        ...    t.update(len(current_buffer))
        >>> t.close()
        The last line is highly recommended, but possibly not necessary if
        `t.update()` will be called in such a way that `filesize` will be
        exactly reached and printed.

        Parameters
        ----------
        n  : int
            Increment to add to the internal counter of iterations
            [default: 1].
        """
        if self.disable:
            return

        if n < 0:
            raise ValueError("n ({0}) cannot be negative".format(n))
        self.n += n

        delta_it = self.n - self.last_print_n  # should be n?
        if delta_it >= self.miniters:
            # We check the counter first, to reduce the overhead of time()
            cur_t = self._time()
            delta_t = cur_t - self.last_print_t
            if delta_t >= self.mininterval:
                elapsed = cur_t - self.start_t
                # EMA (not just overall average)
                if self.smoothing and delta_t:
                    self.avg_time = delta_t / delta_it \
                        if self.avg_time is None \
                        else self.smoothing * delta_t / delta_it + \
                        (1 - self.smoothing) * self.avg_time

                if not hasattr(self, "sp"):
                    raise TqdmDeprecationWarning("""\
Please use `tqdm_gui(...)` instead of `tqdm(..., gui=True)`
""", fp_write=getattr(self.fp, 'write', sys.stderr.write))

                if self.pos:
                    self.moveto(self.pos)

                # Print bar's update
                self.sp(self.format_meter(
                    self.n, self.total, elapsed,
                    (self.dynamic_ncols(self.fp) if self.dynamic_ncols
                     else self.ncols),
                    self.desc, self.ascii, self.unit, self.unit_scale,
                    1 / self.avg_time if self.avg_time else None,
                    self.bar_format))

                if self.pos:
                    self.moveto(-self.pos)

                # If no `miniters` was specified, adjust automatically to the
                # maximum iteration rate seen so far.
                # e.g.: After running `tqdm.update(5)`, subsequent
                # calls to `tqdm.update()` will only cause an update after
                # at least 5 more iterations.
                if self.dynamic_miniters:
                    if self.maxinterval and delta_t > self.maxinterval:
                        self.miniters = self.miniters * self.maxinterval \
                            / delta_t
                    elif self.mininterval and delta_t:
                        self.miniters = self.smoothing * delta_it \
                            * self.mininterval / delta_t + \
                            (1 - self.smoothing) * self.miniters
                    else:
                        self.miniters = self.smoothing * delta_it + \
                            (1 - self.smoothing) * self.miniters

                # Store old values for next call
                self.last_print_n = self.n
                self.last_print_t = cur_t

    def close(self):
    def unpause(self):
    def set_description(self, desc=None):
    def moveto(self, n):
    def clear(self, nomove=False):
    def refresh(self):
*/
