// g++ -std=c++14 -O3

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

constexpr int N = 1024;

using Set = bitset<N>;

template <typename T>
inline ostream& operator<<(ostream& o, const vector<T>& a) {
  for (auto i = a.begin(); i != a.end(); ++i)
    o << (i == a.begin() ? "{" : ", ") << int(*i);
  if (a.empty()) return o << "{}";
  return o << "}";
}

inline ostream& operator<<(ostream& o, const bitset<N>& a) {
  bool first = true;
  for (int i = 0; i < N; ++i)
    if (a[i]) o << (first ? (first = false, "{") : ", ") << i;
  if (first) return o << "{}";
  return o << "}";
}

class Teller {
 public:
  Teller(const vector<int>& truth, bool verbose = false)
      : truth(truth), verbose(verbose) {
    if (verbose) cout << "  truth = " << truth << endl;
  }

  vector<bool> ask(const vector<Set>& q) {
    assert(!answered);  // Cannot ask after submit_answer.
    rounds += 1;
    exams += q.size();
    vector<bool> response;
    response.reserve(q.size());
    for (const auto& p : q) {
      bool has = 0;
      for (int i : truth) has |= p[i];
      response.push_back(has);
    }
    if (verbose) {
      cout << "  Round " << rounds << ": asked (size = " << q.size() << "):\n";
      for (const auto& p : q) cout << "    " << p << endl;
      cout << "    result = " << response << ".\n";
    }
    return response;
  }

  bool submit_answer(const vector<int>& ans) {
    assert(!answered);  // Cannot submit_answer again.
    answered = true;
    bool correct = (ans == truth);
    if (verbose) {
      cout << "  Used " << rounds << " round(s) and " << exams << " exam(s).\n";
      cout << "  " << (correct ? "Correct" : "Wrong") << " answer " << ans
           << ".\n";
    }
    return correct;
  }

  bool is_answered() { return answered; }
  int get_rounds() { return rounds; }
  int get_exams() { return exams; }

 private:
  vector<int> truth;
  int rounds = 0, exams = 0;
  bool answered = false, verbose;
};

class Strategy {
 public:
  // Name to be displayed.
  virtual string name() const { return "UnnamedStrategy"; }

  // Whether this strategy can run with n.
  virtual bool support(int n) const = 0;

  // Maximum number of rounds needed in the worst case with n.
  virtual int worst_rounds(int n) const = 0;

  // Maximum number of exams needed in the worst case with n.
  virtual int worst_exams(int n) const = 0;

  // Executes the strategy with n and the teller agent.
  virtual bool run(int n, Teller* teller) const = 0;
};

class OneRoundStrategy : public Strategy {
 protected:
  // For an one-round strategy, all it needs to give is a query.
  virtual vector<Set> make_query(int n) const = 0;

 public:
  int worst_rounds(int n) const final { return 1; };
  int worst_exams(int n) const final { return get_query(n).size(); };

  // The default run method using naive enumeration.
  // Subclass may override to provide a faster solution.
  virtual vector<int> run(int n, const vector<bool>& response) const {
    if (try_answer(n, {}) == response) return {};
    vector<int> possible;
    for (int i = 0; i < n; ++i) {
      auto r = try_answer(n, {i});
      bool ok = true;
      for (int k = 0; k < r.size(); ++k)
        if (r[k] && !response[k]) {
          ok = false;
          break;
        }
      if (!ok) continue;
      if (r == response) return {i};
      possible.push_back(i);
    }
    for (int j : possible)
      for (int i : possible) {
        if (i >= j) break;
        if (try_answer(n, {i, j}) == response) return {i, j};
      }
    return {};  // not found
  }

  const vector<Set>& get_query(int n) const {
    if (!q_cache.count(n)) q_cache[n] = make_query(n);
    return q_cache[n];
  }

  vector<bool> try_answer(int n, const vector<int>& ans) const {
    vector<bool> response;
    const auto& q = get_query(n);
    response.reserve(q.size());
    for (const auto& p : q) {
      bool has = 0;
      for (int i : ans) has |= p[i];
      response.push_back(has);
    }
    return response;
  }

  bool run(int n, Teller* teller) const final {
    auto ans = run(n, teller->ask(get_query(n)));
    return teller->submit_answer(ans);
  }

 private:
  mutable unordered_map<int, vector<Set>> q_cache;
};

inline void print_passed_info(int n, int max_rounds, int max_exams,
                              const Strategy& strategy) {
  cout << "  Passed! worst rounds = " << max_rounds
       << ", worst exams = " << max_exams << ".\n";
  if (max_rounds != strategy.worst_rounds(n))
    cout << "    strategy.worst_rounds(" << n
         << ") = " << strategy.worst_rounds(n)
         << ", but actual worst rounds = " << max_rounds << ".\n";
  if (max_exams != strategy.worst_exams(n))
    cout << "    strategy.worst_exams(" << n
         << ") = " << strategy.worst_exams(n)
         << ", but actual worst exams = " << max_exams << ".\n";
}

inline void log_progress(int64_t x, int64_t y, int64_t n) {
  static int ct = 0;
  if (++ct % 4096 == 0) {
    double progress = ((y) * (y + 1) + x * 2 + 4.) / (n * (n + 1) + 2);
    printf("  %.2lf%%..\r", progress * 100);
  }
}

bool brute_force_verify(int n, const Strategy& strategy) {
  cout << "Verifying " << strategy.name() << " with n = " << n << ".\n";
  if (!strategy.support(n)) {
    cout << "  Failed! N not supported.\n";
    return false;
  }
  int max_rounds = 0, max_exams = 0;
  auto verify_case = [&](const vector<int>& truth) {
    Teller teller(truth);
    if (!strategy.run(n, &teller) || !teller.is_answered()) {
      cout << "  Failed!  \n";
      Teller verbose_teller(truth, true);
      strategy.run(n, &verbose_teller);
      if (!verbose_teller.is_answered())
        cout << "  Strategy did not submit an answer.\n";
      cout << "  truth = " << truth << ".\n";
      return false;
    }
    max_rounds = max(max_rounds, teller.get_rounds());
    max_exams = max(max_exams, teller.get_exams());
    return true;
  };

  if (!verify_case({})) return false;
  for (int y = 0; y < n; ++y) {
    if (!verify_case({y})) return false;
    for (int x = 0; x < y; ++x) {
      if (!verify_case({x, y})) return false;
      log_progress(x, y, n);
    }
  }
  print_passed_info(n, max_rounds, max_exams, strategy);
  return true;
}

bool disjoint_verify(int n, const OneRoundStrategy& strategy,
                     bool verbose = true) {
  if (verbose) {
    cout << "Verifying " << strategy.name() << " with n = " << n << ".\n";
    if (!strategy.support(n)) {
      cout << "  Failed! Not supported.\n";
      return false;
    }
  }
  unordered_map<vector<bool>, vector<int>> m;
  int max_rounds = 1, max_exams = strategy.worst_exams(n);
  m[strategy.try_answer(n, {})] = {};
  auto verify_case = [&](const vector<int>& truth) {
    auto r = strategy.try_answer(n, truth);
    if (m.count(r)) {
      if (verbose)
        cout << "  Failed! Cannot distinguish " << m[r] << " and " << truth
             << ".\n";
      return false;
    }
    m[r] = truth;
    return true;
  };
  for (int y = 0; y < n; ++y) {
    if (!verify_case({y})) return false;
    for (int x = 0; x < y; ++x) {
      if (!verify_case({x, y})) return false;
      if (verbose) log_progress(x, y, n);
    }
  }
  if (verbose) print_passed_info(n, max_rounds, max_exams, strategy);
  return true;
}

//====================
// One-Round Strategies
//====================

// 1 round, n exams
class NaiveOneRoundStrategy final : public OneRoundStrategy {
 public:
  string name() const final { return "NaiveOneRoundStrategy"; }
  bool support(int n) const final { return n >= 0; }

 private:
  vector<Set> make_query(int n) const final {
    vector<Set> q(n);
    for (int i = 0; i < n; ++i) q[i][i] = true;
    return q;
  }
};

// credit: Zhengjie Miao
// 1 round, O(log^2 N) exams
class DigitChecksumOneRoundStrategy : public OneRoundStrategy {
 public:
  DigitChecksumOneRoundStrategy(int base, int len) : base(base), len(len) {}
  string name() const override {
    return "DigitChecksumOneRoundStrategy(" + to_string(base) + "," +
           to_string(len) + ")";
  }
  bool support(int n) const override { return 0 <= n && n <= pow(base, len); }

  vector<int> run(int n, const vector<bool>& r) const override {
    if (find(r.begin(), r.end(), true) == r.end()) return {};
    vector<vector<int>> c(len);
    vector<int> cs(len);
    int cmask = 0;
    for (int i = 0; i < len; ++i) {
      for (int v = 0; v < base; ++v)
        if (r[pos_digit_value(i, v)]) c[i].push_back(v);
      if (c[i].size() > 2 || c[i].empty()) return {};  // cannot solve
      cs[i] = c[i].size() == 1 ? c[i][0] * 2 : c[i][0] + c[i][1];
      cmask |= (c[i].size() == 1) << i;
    }
    for (int ci = 0; ci < (1 << len); ++ci)
      if (!(ci & cmask)) {
        int x = 0, y = 0;
        for (int i = len - 1; i >= 0; --i) {
          int xd = c[i][ci >> i & 1];
          x = x * base + xd;
          y = y * base + cs[i] - xd;
        }
        if (x > y) continue;
        vector<bool> nr(r.size());
        auto dx = digits(x), dy = digits(y);
        for (int i = 0; i < len; ++i) {
          nr[pos_digit_value(i, dx[i])] = true;
          nr[pos_digit_value(i, dy[i])] = true;
          for (int j = i + 1; j < len; ++j) {
            nr[pos_pair_checksum(i, j, (dx[i] + dx[j]) % base)] = true;
            nr[pos_pair_checksum(i, j, (dy[i] + dy[j]) % base)] = true;
          }
        }
        if (nr == r) return x == y ? vector<int>{x} : vector<int>{x, y};
      }
    return {};  // Answer not found.
  }

 private:
  vector<int> digits(int k) const {
    vector<int> d(len);
    for (int i = 0; i < len; ++i) {
      d[i] = k % base;
      k /= base;
    }
    return d;
  }

  int pos_digit_value(int d, int v) const {
    return v * len * (len + 1) / 2 + d;
  }
  int pos_pair_checksum(int d1, int d2, int v) const {
    return v * len * (len + 1) / 2 + len + (2 * len - d1 - 1) * d1 / 2 + d2 -
           d1 - 1;
  }

  vector<Set> make_query(int n) const override {
    vector<Set> q(base * len * (len + 1) / 2);
    for (int k = 0; k < n; ++k) {
      auto d = digits(k);
      for (int i = 0; i < len; ++i) {
        q[pos_digit_value(i, d[i])][k] = true;
        for (int j = i + 1; j < len; ++j)
          q[pos_pair_checksum(i, j, (d[i] + d[j]) % base)][k] = true;
      }
    }
    return q;
  }

  int base, len;
};

// credit: Changji Xu
// 1 round, O(logN) exams
class RandomOneRoundStrategy : public OneRoundStrategy {
 public:
  RandomOneRoundStrategy(int max_n, int threshold, int max_q, int64_t rand_seed)
      : max_n(max_n),
        threshold(threshold),
        max_q(max_q),
        rand_seed(rand_seed) {}

  string name() const override {
    return "RandomOneRoundStrategy(" + to_string(max_n) + "," +
           to_string(threshold) + "," + to_string(max_q) + "," +
           to_string(rand_seed) + ")";
  }
  bool support(int n) const override { return 0 <= n && n <= max_n; }

 private:
  vector<Set> make_query(int n) const override {
    vector<Set> q(max_q);
    auto gen = mt19937_64(rand_seed);
    auto rnd = uniform_int_distribution<int>(0, max_n - 1);
    for (auto& p : q)
      for (int i = 0; i < max_n; ++i)
        if (rnd(gen) < threshold && i < n) p[i] = true;
    return q;
  }

  int max_n, threshold, max_q;
  int64_t rand_seed;
};

// credit: Pufan He
// 1 round, O(logN^(1.58)) exams
class RecursiveOneRoundStrategy : public OneRoundStrategy {
 public:
  RecursiveOneRoundStrategy() {}

  string name() const override { return "RecursiveOneRoundStrategy"; }
  bool support(int n) const override { return n >= 0; }

 private:
  vector<Set> make_query(int n) const override {
    if (n < 7) return naive_strategy.get_query(n);
    int m = sqrt(n);
    while (m * m < n) ++m;
    vector<Set> q;
    for (const auto& p : get_query(m)) {
      Set sx = 0, sy = 0;
      for (int j = 0; j < m; ++j)
        if (p[j])
          for (int i = 0; i < m; ++i) {
            if (i * m + j < n) sx[i * m + j] = true;
            if (j * m + i < n) sy[j * m + i] = true;
          }
      q.emplace_back(move(sx));
      q.emplace_back(move(sy));
    }
    for (const auto& p : get_query(m - 2)) {
      Set sz = 0;
      for (int j = 0; j < m - 2; ++j)
        if (p[j])
          for (int i = 0; i < m; ++i) {
            int y = (j + m - i) % m;
            if (i * m + y < n) sz[i * m + y] = true;
          }
      q.emplace_back(move(sz));
    }
    return q;
  }

 private:
  NaiveOneRoundStrategy naive_strategy;
};

//====================
// Two-Round Strategies
//====================

// credit: Pufan He
// 2 rounds, <3*log_2(N) exams
class TwoRoundStrategy : public Strategy {
 public:
  string name() const override { return "TwoRoundStrategy"; }
  bool support(int n) const override { return n >= 0; }
  int worst_rounds(int n) const override { return 2; }
  int worst_exams(int n) const override {
    return get_worst_by_pivot(n, find_pivot(n));
  }

  bool run(int n, Teller* teller) const override {
    if (!n) return teller->submit_answer({});
    int p = find_pivot(n);
    vector<Set> q;
    for (int i = 0; i < p; ++i) {
      Set s0 = 0, s1 = 0;
      for (int k = 0; k < n; ++k)
        if (k >> i & 1)
          s1[k] = true;
        else
          s0[k] = true;
      q.push_back(s0);
      q.push_back(s1);
    }
    auto r = teller->ask(q);
    int common = 0, diff = 0;
    for (int i = 0; i < p; ++i) {
      if (!r[i * 2] && !r[i * 2 + 1]) return teller->submit_answer({});
      if (r[i * 2] && r[i * 2 + 1])
        diff |= 1 << i;
      else
        common |= !r[i * 2] << i;
    }
    q.clear();
    if (diff) {
      int d = 0;
      while (!(diff >> d & 1)) ++d;
      for (int i = p; n >> i; ++i) {
        Set s = 0;
        for (int k = 0; k < n; ++k)
          if (!(k >> d & 1) && (k >> i & 1)) s[k] = true;
        q.push_back(s);
      }
      int x = common, y = common | 1 << d, ri = q.size();
      for (int i = 0; n >> i; ++i) {
        Set s = 0;
        for (int k = 0; k < n; ++k)
          if ((k >> d & 1) && (k >> i & 1)) s[k] = true;
        q.push_back(s);
      }
      r = teller->ask(q);
      for (int i = 0; n >> i; ++i) {
        y |= r[ri] << i;
        ++ri;
      }
      x = (diff ^ y) & ((1 << p) - 1);
      ri = 0;
      for (int i = p; n >> i; ++i) {
        x |= r[ri] << i;
        ++ri;
      }
      if (x > y) swap(x, y);
      return teller->submit_answer({x, y});
    } else {
      int n2 = ((n - 1) >> p) + 1;
      auto q2 = one_round_strategy.get_query(n2);
      q.resize(q2.size(), 0);
      for (int i = 0; i < q2.size(); ++i) {
        for (int j = 0; j < n2; ++j)
          if (q2[i][j])
            for (int k = 0; k < (1 << p); ++k) q[i][j << p | k] = true;
      }
      r = teller->ask(q);
      auto one_round_answer = one_round_strategy.run(n2, r);
      vector<int> answer;
      for (int i : one_round_answer) answer.push_back(i << p | common);
      return teller->submit_answer(answer);
    }
  }

 private:
  int find_pivot(int n) const {
    int l = 0;
    while (n >> l) ++l;
    int best = l, q = 3 * l;
    for (int p = 0; p < l; ++p) {
      int nq = get_worst_by_pivot(n, p);
      if (nq < q) {
        best = p;
        q = nq;
      }
    }
    return best;
  }

  int get_worst_by_pivot(int n, int p) const {
    int l = 0;
    while (n >> l) ++l;
    return p * 2 +
           max(l + l - p, one_round_strategy.worst_exams(((n - 1) >> p) + 1));
  }

  RecursiveOneRoundStrategy one_round_strategy;
};

//====================
// Interactive Strategies
//====================

// credit: Pufan He
// log_2(N) rounds, 2*log_2(N) exams
class InteractiveStrategy : public Strategy {
 public:
  string name() const override { return "InteractiveStrategy"; }
  bool support(int n) const override { return n >= 0; }
  int worst_rounds(int n) const override {
    int i = 1, r = 0;
    while (i < n) i *= 2, ++r;
    return r;
  }
  int worst_exams(int n) const override { return worst_rounds(n) * 2; }

  bool run(int n, Teller* teller) const override {
    if (!n) return teller->submit_answer({});
    int common = 0;
    for (int i = 0; n >> i; ++i) {
      Set s0 = 0, s1 = 0;
      for (int k = 0; k < n; ++k)
        if (k >> i & 1)
          s1[k] = true;
        else
          s0[k] = true;
      auto r = teller->ask({s0, s1});
      if (r[0] && r[1]) {
        int x = common, y = common | 1 << i;
        for (int j = i + 1; n >> j; ++j) {
          Set sx = 0, sy = 0;
          for (int k = 0; k < n; ++k)
            if (k >> j & 1) {
              if (k >> i & 1)
                sy[k] = true;
              else
                sx[k] = true;
            }
          r = teller->ask({sx, sy});
          x |= r[0] << j;
          y |= r[1] << j;
        }
        if (x > y) swap(x, y);
        return teller->submit_answer({x, y});
      }
      if (!r[0] && !r[1]) return teller->submit_answer({});
      common |= !r[0] << i;
    }
    return teller->submit_answer({common});
  }
};

// credit: Pufan He
// <=2log_2(N) rounds and exams
// slightly better than InteractiveStrategy in some Ns.
class BetterInteractiveStrategy : public Strategy {
 public:
  string name() const override { return "BetterInteractiveStrategy"; }
  bool support(int n) const override {
    prepare(n);
    return n >= 0;
  }
  int worst_rounds(int n) const override {
    prepare(n);
    return f012[n];
  }
  int worst_exams(int n) const override { return worst_rounds(n); }

  bool run(int n, Teller* teller) const override {
    prepare(n);
    int l = 0, xr = n, yl = 0, r = n;
    bool y_must_exist = false;
    if (!w012[r - l]) {
      auto c = teller->ask({range(l, r)})[0];
      if (!c) return teller->submit_answer({});
    }

    for (;;) {
      int k = w12[r - l][xr - l];
      if (!k) {
        if (xr == l + 1) yl = xr;
        break;
      }
      if (k < xr - l) {
        auto c = teller->ask({range(l, l + k)})[0];
        if (c)
          xr = l + k;
        else
          yl = l += k;
      } else {
        auto c = teller->ask({range(l + k, r)})[0];
        if (c) {
          yl = l + k;
          y_must_exist = true;
          break;
        }
        r = l + k;
      }
    }
    if (l >= xr || l >= r || (l + 1 == r && !teller->ask({range(l, r)})[0]))
      return teller->submit_answer({});
    int x = binsearch(l, xr, true, teller),
        y = binsearch(yl, r, y_must_exist, teller);
    if (x < 0 && y < 0) return teller->submit_answer({});
    if (x >= 0 && y >= 0) return teller->submit_answer({x, y});
    return teller->submit_answer({x >= 0 ? x : y});
  }

 private:
  static Set range(int l, int r) {
    Set s = 0;
    for (int i = l; i < r; ++i) s[i] = true;
    return s;
  }

  int binsearch(int l, int r, bool must_exist, Teller* teller) const {
    while (l + 1 < r) {
      int m = (l + r + 1) / 2;
      if (teller->ask({range(l, m)})[0])
        r = m, must_exist = true;
      else
        l = m;
    }
    if (l >= r) return -1;
    if (must_exist) return l;
    return teller->ask({range(l, r)})[0] ? l : -1;
  }

  void prepare(int n) const {
    int done = f012.size() - 1;
    if (done >= n) return;
    for (int i = done + 1; i <= n; ++i) {
      f01.push_back(f01[i / 2] + 1);
      f1.push_back(f01[i - 1]);
      vector<int> f(i + 1, 0), g(i + 1, 0);
      f[0] = f01[i];
      f[1] = f01[i - 1];
      for (int j = 2; j <= i; ++j) {
        f[j] = i + 1;
        for (int k = 1; k < i; ++k) {
          int c0 = k < j ? f12[i - k][j - k] : f12[k][j];
          int c1 = k < j ? f[k] : f1[i - k] + f1[j];
          if (1 + max(c0, c1) < f[j]) {
            f[j] = 1 + max(c0, c1);
            g[j] = k;
          }
        }
      }
      f12.emplace_back(move(f));
      w12.emplace_back(move(g));
      f012.push_back(1 + f12[i][i]);
      w012.push_back(0);
      for (int k = 1; k < i; ++k) {
        int c0 = f012[i - k];
        int c1 = f12[i][k];
        if (1 + max(c0, c1) < f012[i]) {
          f012[i] = 1 + max(c0, c1);
          w012[i] = k;
        }
      }
    }
  }

  mutable vector<int> f01 = {0}, f1 = {0}, f012 = {0}, w012 = {0};
  mutable vector<vector<int>> f12 = {{0}}, w12 = {{0}};
};

void run_random(int n, int best_exams = -1, int64_t best_seed = -1,
                int64_t tested_max_seed = -1) {
  if (best_exams >= 0) {
    auto s = RandomOneRoundStrategy(n, n / 3, best_exams, best_seed);
    for (auto& p : s.get_query(n)) cout << p << endl;
    assert(disjoint_verify(n, s));
  } else {
    best_exams = n;  // naive will do n
  }

  // look for the next better rand seed
  for (auto seed = tested_max_seed + 1;;) {
    auto s = RandomOneRoundStrategy(n, n / 3, best_exams - 1, seed);
    if (disjoint_verify(n, s, false)) {
      --best_exams, best_seed = seed;
      for (auto& p : s.get_query(n)) cout << p << endl;
      cout << "New best exams & seed: " << best_exams << ", " << best_seed
           << ".\n";
    } else {
      if (seed % 16384 == 0) cout << "verified seeds up to " << seed << "..\r";
      ++seed;
    }
  }
}

void state_of_the_art() {
  int n = 1000;

  // Best known interactive strategy for 1000.
  assert(brute_force_verify(n, BetterInteractiveStrategy()));

  // Best known two-round strategy for 1000.
  assert(brute_force_verify(n, TwoRoundStrategy()));

  // Best known one-round strategy for 1000.
  assert(disjoint_verify(n, RecursiveOneRoundStrategy()));

  // Past record holders and interesting experiments.
  if (0) {
    assert(brute_force_verify(860, BetterInteractiveStrategy()));
    assert(disjoint_verify(1000, DigitChecksumOneRoundStrategy(10, 3)));
    assert(disjoint_verify(1024, DigitChecksumOneRoundStrategy(4, 5)));
    assert(brute_force_verify(1000, InteractiveStrategy()));
    int best_rand_exams = 50;
    int64_t best_rand_seed = 42337;
    // int64_t tested_max_rand_seed = 1189177;
    auto s = RandomOneRoundStrategy(1000, 333, best_rand_exams, best_rand_seed);
    assert(disjoint_verify(1000, s));
  }
}

int main() {
  state_of_the_art();
  return 0;
}
