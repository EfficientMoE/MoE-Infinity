#pragma once

#include <cassert>      // assert
#include <cstddef>      // ptrdiff_t, size_t
#include <unistd.h>     // STDERR_FILENO
#include <iterator>     // iterator
#include <type_traits>  // is_pointer, ...
#include <atomic>       // atomic
#include <cstring>      // strlen
#include <cerrno>       // EAGAIN
#include <poll.h>       // poll

/** TODO: port from python
 * colorama win
 * weakset
 * _is_utf(encoding)
 * _is_ascii(s)
 * _supports_unicode(file)
 * _environ_cols()
 * _sh(const char *cmd[], ...)
 */

#ifndef constexpr
  #define constexpr static const
#endif

#ifndef explicit
  #define explicit
#endif

#ifndef noexcept
  #define noexcept
#endif

namespace tqdm {

template <typename _Iterator>
/**
Wrapper for pointers and std container iterators.
@author Casper da Costa-Luis
*/
class MyIteratorWrapper
    : public std::iterator<
          std::forward_iterator_tag,
          typename std::iterator_traits<_Iterator>::value_type> {
  mutable _Iterator p;  // TODO: remove this mutable

 public:
  // already done by std::iterator
  typedef typename std::iterator_traits<_Iterator>::value_type value_type;

  explicit MyIteratorWrapper(_Iterator x) : p(x) {}
  // default construct gives end
  MyIteratorWrapper() : p(nullptr) {}
  explicit MyIteratorWrapper(const MyIteratorWrapper& mit) : p(mit.p) {}

  // override this in Tqdm class
  virtual void _incr() { ++p; }
  // override this in Tqdm class
  virtual void _incr() const { ++p; }

  MyIteratorWrapper& operator++() {
    // assert(this->bool() && "Out-of-bounds iterator increment");
    _incr();
    return *this;
  }
  const MyIteratorWrapper& operator++() const {
    _incr();
    return *this;
  }
  MyIteratorWrapper operator++(int) const {
    MyIteratorWrapper tmp(*this);
    _incr();
    return tmp;
  }
  template <class Other>
  // two-way comparison: v.begin() == v.cbegin() and vice versa
  bool operator==(const MyIteratorWrapper<Other>& rhs) const {
    return p == rhs.p;
  }
  template <class Other>
  bool operator!=(const MyIteratorWrapper<Other>& rhs) const {
    return p != rhs.p;
  }
  template <class Other>
  size_t operator-(const MyIteratorWrapper<Other>& rhs) {
    return p - rhs.p;
  }
  // template <typename = typename std::enable_if<
  //               !std::is_const<value_type>::value>::type>
  value_type& operator*() {
    // assert(this->bool() && "Invalid iterator dereference!");
    return *p;
  }
  const value_type& operator*() const {
    // assert(this->bool() && "Invalid iterator dereference!");
    return *p;
  }
  // template <typename = typename std::enable_if<
  //               !std::is_const<value_type>::value>::type>
  value_type& operator->() {
    // assert(this->bool() && "Invalid iterator dereference!");
    return *p;
  }
  const value_type& operator->() const {
    // assert(this->bool() && "Invalid iterator dereference!");
    return *p;
  }
  // @return the underlying iterator
  _Iterator& get() { return p; }
  const _Iterator& get() const { return p; }
  // TODO: const _Iterator &get() const { return p; }, etc ...
  //
  void swap(MyIteratorWrapper& other) noexcept { std::swap(p, other.p); }

  // One way conversion: iterator -> const_iterator
  // template <typename =
  //               typename
  //               std::enable_if<!std::is_const<_Iterator>::value>::type>
  // operator MyIteratorWrapper<typename std::add_const<_Iterator>::type>()
  // const {
  //   return MyIteratorWrapper<const _Iterator>(p);
  // }
  template <typename = typename std::is_pointer<_Iterator>>
  explicit operator bool() const {
    return p != nullptr;
  }
};

template <typename _Iterator,
          typename _MyIteratorWrapper = MyIteratorWrapper<_Iterator>>
_MyIteratorWrapper myIteratorWrapper(_Iterator x) {
  return _MyIteratorWrapper(x);
}

template <typename IntType = int>
class RangeIterator : public std::iterator<std::forward_iterator_tag, IntType> {
 private:
  mutable IntType current;
  IntType total;
  IntType step;

 public:
  RangeIterator(IntType total) : current(0), total(total), step(1) {}
  RangeIterator(IntType start, IntType total)
      : current(start), total(total), step(1) {}
  RangeIterator(IntType start, IntType total, IntType step)
      : current(start), total(total), step(step) {}
  IntType& operator*() { return current; }
  const IntType& operator*() const { return current; }
  RangeIterator& operator++() {
    current += step;
    return *this;
  }
  const RangeIterator& operator++() const {
    current += step;
    return *this;
  }
  RangeIterator operator++(int) const {
    RangeIterator tmp(*this);
    operator++();
    return tmp;
  }
  explicit operator bool() const { return current < total; }
  size_t size_remaining() const { return (total - current) / step; }

  /** here be dragons */

  // only use as (it != end), not as (end != it)
  bool operator!=(const RangeIterator&) const { return current < total; }
  bool operator==(const RangeIterator&) const { return current >= total; }
  IntType operator-(const RangeIterator& it) const {
    // it's used in `end - begin`, but `end` is only a sentinel
    // so let's use `begin `to be consistent
    return it.size_remaining();
  }
};

const char* _term_move_up() {
  return
#if defined(IS_WIN) && !defined(colorama)
      ""
#else
      "\x1b[A"
#endif
      ;
}

static void wait_for_write(int fd) {
  struct pollfd pfd;
  pfd.fd = fd;
  pfd.events = POLLOUT;
  (void)::poll(&pfd, 1, -1);
}

// Write a buffer fully or not at all.
// If false is returned, caller may check errno to see if it's EAGAIN
// or a real error.
bool write_harder(int fd, const char* buf, size_t len) {
  bool did_anything = false;

  while (len) {
    ssize_t res = ::write(fd, buf, len);
    if (res == -1) {
      if (errno == EAGAIN) {
        if (!did_anything) {
          return false;
        }
        wait_for_write(fd);
        continue;
      }
      return false;
    }
    assert(res != 0);
    did_anything = true;
    buf += res;
    len -= res;
  }
  return true;
}

class AbstractLine;

template <class Node>
class AtomicList;

// CRTP
template <class Node>
class AtomicNode {
  friend class AtomicList<Node>;

  std::atomic<Node*> intrusive_link_next;
  std::atomic<Node*> intrusive_link_prev;

  AtomicNode(Node* next, Node* prev);

 public:
  // Node is initially unattached
  AtomicNode();
  ~AtomicNode();
};

// A non-owning intrusive linked list,
// using atomics to ensure thread- and signal- safety.
template <class Node>
class AtomicList {
  AtomicNode<Node> meta;

 public:
  AtomicList();
  ~AtomicList();

  void append(Node* node);
};

template <class Node>
AtomicNode<Node>::AtomicNode(Node* next, Node* prev) {
  intrusive_link_next.store(next);
  intrusive_link_prev.store(prev);
}
template <class Node>
AtomicNode<Node>::AtomicNode() {
  intrusive_link_next.store(nullptr);
  intrusive_link_prev.store(nullptr);
}
template <class Node>
AtomicNode<Node>::~AtomicNode() {}

class AbstractLine : public AtomicNode<AbstractLine> {
  friend class Sink;

  struct {
    bool dirty : 1;
  } flags;

 public:
  AbstractLine() : flags{} {}
  // Due to how vtables work, it is cheaper to *not* inline this.
  virtual ~AbstractLine() {};

  virtual void write(int fd) = 0;

 protected:
  void not_dirty() { this->flags.dirty = false; }
};

class StaticTextLine : public AbstractLine {
  const char* text;

 public:
  template <size_t n>
  StaticTextLine(const char (&lit)[n]) : text(lit) {}
  void write(int fd) override {
    bool ok = write_harder(fd, this->text, strlen(this->text));
    if (ok) this->not_dirty();
  }
};

struct SinkOptions {
  // Only mandatory field. Everything else can just be zeroed.
  int fd;

  int tty_width;
  int tty_height;

  // Additional options will be added in future.
  SinkOptions(int fd) : fd(fd) {};
};

class Sink;
// We do still need a global list of sinks in order to handle signals.
// This is still a win over making a single global list of AbstractLine
// instances, since we can skip entirely any Sink which does not express
// interest in asynchronous updates.
static AtomicList<Sink> all_sinks;

class Sink : public AtomicNode<Sink> {
  SinkOptions opts;
  AtomicList<AbstractLine> lines;

 public:
  explicit Sink(SinkOptions o) : opts(o) { all_sinks.append(this); }
  Sink(Sink&&) = delete;
  Sink& operator=(Sink&&) = delete;
};

Sink standard_sink(SinkOptions(STDERR_FILENO));

// static void wait_for_write(int fd);

// Write a buffer fully or not at all.
// If false is returned, caller may check errno to see if it's EAGAIN
// or a real error.
bool write_harder(int fd, const char* buf, size_t len);

// To more easily maintain the doubly-linked structure, loop to itself
// rather than using NULL pointers.
template <class Node>
AtomicList<Node>::AtomicList()
    : meta(static_cast<Node*>(&meta), static_cast<Node*>(&meta)) {}

// Nothing to do - we didn't allocate any objects, merely borrow.
template <class Node>
AtomicList<Node>::~AtomicList() {
  // TODO: We *really* shouldn't get here with a non-empty node set.
  // Should we set all nodes to NULL to indicate they have?
  // Otherwise we're stuck with dangling pointers in the edges ...
}

template <class Node>
void AtomicList<Node>::append(Node* node) {
  (void)node;
#if 0
  Node *singular = &meta;
  // TODO when deletion is added, need to disable that
  // for the duration of this function (using a spinlock).
  // But, if the spinlock is currently held by the main thread,
  // and we are a signal handler, we need to advance anyway.
  // I think our spinlocks need to be recursive, which means tracking
  // the thread ID.
  //
  // Actually, we might not need to disable it, just make sure it
  // always points to a later node, perhaps? But what if there
  // is no later node? Maybe use the list-itself-is-a-node trick?
  assert (node->intrusive_link_next.load() == nullptr);
  assert (node->intrusive_link_prev.load() == nullptr);
  // Note that tail is a pointer to a (usually) NULL pointer.
  std::atomic<Node *> *tail = this->approx_tail;
  assert (tail != nullptr);
  while (true)
  {
      // The tail may have moved, if someone else is also appending.
      while (Node *tmp = tail->load())
      {
          tail = &tmp->intrusive_link_next;
      }
      // First argument is a reference, but we can't reuse
      // the new value, because we want to advance.
      Node *expected = nullptr;
      if (tail->compare_exchange_weak(expected, node))
      {
          break;
      }
  }
  // If we're wrong, nobody cares until the next append,
  // which will fix this anyway.
  this->approx_tail = &node->intrusive_link_next;
#endif
}

}  // namespace tqdm
