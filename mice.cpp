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
    quests += q.size();
    vector<bool> response;
    response.reserve(q.size());
    for (const auto& p : q) {
      bool has = 0;
      for (int i : truth) has |= p[i];
      response.push_back(has);
    }
    if (verbose) {
      cout << "    Round " << rounds << ": asked (size = " << q.size()
           << "):\n";
      for (const auto& p : q) cout << "    " << p << endl;
      cout << "      Responded " << response << ".\n";
    }
    return response;
  }

  bool submit_answer(const vector<int>& ans) {
    assert(!answered);  // Cannot submit_answer again.
    answered = true;
    bool correct = (ans == truth);
    if (verbose) {
      cout << "  " << (correct ? "Correct" : "Wrong") << " answer " << ans
           << ".\n";
      cout << "  Used " << rounds << " round(s) and " << quests
           << " quest(s).\n";
    }
    return correct;
  }

  bool is_answered() { return answered; }
  int get_rounds() { return rounds; }
  int get_quests() { return quests; }

 private:
  vector<int> truth;
  int rounds = 0, quests = 0;
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

  // Maximum number of quests needed in the worst case with n.
  virtual int worst_quests(int n) const = 0;

  // Executes the strategy with n and the teller agent.
  virtual bool run(int n, Teller* teller) const = 0;
};

class OneRoundStrategy : public Strategy {
 protected:
  // For an one-round strategy, all it needs to give is a query.
  virtual vector<Set> make_query(int n) const = 0;

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

 public:
  int worst_rounds(int n) const final { return 1; };
  int worst_quests(int n) const final { return get_query(n).size(); };

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

inline void print_passed_info(int n, int max_rounds, int max_quests,
                              const Strategy& strategy) {
  cout << "  Passed! worst rounds = " << max_rounds
       << ", worst quests = " << max_quests << ".\n";
  if (max_rounds != strategy.worst_rounds(n))
    cout << "    strategy.worst_rounds(" << n
         << ") = " << strategy.worst_rounds(n)
         << ", but actual worst rounds = " << max_rounds << ".\n";
  if (max_quests != strategy.worst_quests(n))
    cout << "    strategy.worst_quests(" << n
         << ") = " << strategy.worst_quests(n)
         << ", but actual worst quests = " << max_quests << ".\n";
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
  int max_rounds = 0, max_quests = 0;
  auto verify_case = [&](const vector<int>& truth) {
    Teller teller(truth);
    if (!strategy.run(n, &teller) || !teller.is_answered()) {
      cout << "  Failed!          \n";
      Teller verbose_teller(truth, true);
      strategy.run(n, &verbose_teller);
      if (!verbose_teller.is_answered())
        cout << "  Strategy did not submit an answer.\n";
      cout << "  But truth = " << truth << ".\n";
      return false;
    }
    max_rounds = max(max_rounds, teller.get_rounds());
    max_quests = max(max_quests, teller.get_quests());
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
  print_passed_info(n, max_rounds, max_quests, strategy);
  return true;
}

bool disjoint_verify(int n, const OneRoundStrategy& strategy) {
  cout << "Verifying " << strategy.name() << " with n = " << n << ".\n";
  if (!strategy.support(n)) {
    cout << "  Failed! Not supported.\n";
    return false;
  }
  unordered_map<vector<bool>, vector<int>> m;
  int max_rounds = 1, max_quests = strategy.worst_quests(n);
  m[strategy.try_answer(n, {})] = {};
  auto verify_case = [&](const vector<int>& truth) {
    auto r = strategy.try_answer(n, truth);
    if (m.count(r)) {
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
      log_progress(x, y, n);
    }
  }
  print_passed_info(n, max_rounds, max_quests, strategy);
  return true;
}

//====================
// Strategies
//====================

// credit: Pufan He
// log_2(N) rounds, 2*log_2(N) quests
class InteractiveStrategy : public Strategy {
 public:
  string name() const override { return "InteractiveStrategy"; }
  bool support(int n) const override { return n >= 0; }
  int worst_rounds(int n) const override {
    int i = 1, r = 0;
    while (i < n) i *= 2, ++r;
    return r;
  }
  int worst_quests(int n) const override { return worst_rounds(n) * 2; }

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
// 2 rounds, <3*log_2(N) quests
class TwoRoundStrategy : public Strategy {
 public:
  string name() const override { return "TwoRoundStrategy"; }
  bool support(int n) const override { return n >= 0; }
  int worst_rounds(int n) const override { return 2; }
  int worst_quests(int n) const override {
    int l = 0;
    while ((1 << l) < n) ++l;
    int p = find_pivot(n);
    return p * 2 + max(l + l - p, 1 << (l - p));
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
      // TODO: instead of this, reuse the best one-round strategy for n = n2.
      q.resize(n2, 0);
      for (int i = 0; i < n2; ++i)
        for (int k = 0; k < (1 << p); ++k) q[i][k | i << p] = true;
      r = teller->ask(q);
      vector<int> c;
      for (int i = 0; i < n2; ++i)
        if (r[i]) c.push_back(common | i << p);
      return teller->submit_answer(c);
    }
  }

 private:
  int find_pivot(int n) const {
    int l = 0;
    while ((1 << l) < n) ++l;
    int best = l, q = 3 * l;
    for (int i = 0; i < l; ++i) {
      int nq = i * 2 + max(l + l - i, 1 << (l - i));
      if (nq < q) {
        best = i;
        q = nq;
      }
    }
    return best;
  }
};

// credit: Zhengjie Miao
// 1 round, O(log^2 N) quests
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
// 1 round, O(logN) quests
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
    for (auto& p : q) {
      for (int i = 0; i < n; ++i)
        if (rnd(gen) < threshold) p[i] = true;
    }
    return q;
  }

  int max_n, threshold, max_q;
  int64_t rand_seed;
};

int main() {
  if (1) {
    auto s = DigitChecksumOneRoundStrategy(10, 3);
    assert(disjoint_verify(1000, s));
  }

  if (1) {
    auto s = DigitChecksumOneRoundStrategy(4, 5);
    assert(disjoint_verify(1024, s));
  }

  if (1) {
    auto s = InteractiveStrategy();
    assert(brute_force_verify(1000, s));
  }

  if (1) {
    auto s = TwoRoundStrategy();
    assert(brute_force_verify(1000, s));
  }

  int best_quests = 51;
  int64_t best_seed = 7397, tested_max_seed = 13000;
  if (1) {
    auto s = RandomOneRoundStrategy(1000, 333, best_quests, best_seed);
    assert(disjoint_verify(1000, s));
  }

  if (1) {
    // look for the next better rand seed
    for (auto seed = tested_max_seed + 1;;) {
      auto s = RandomOneRoundStrategy(1000, 333, best_quests - 1, seed);
      bool success = disjoint_verify(1000, s);
      if (success) {
        --best_quests, best_seed = seed;
      } else {
        ++seed;
      }
      cout << "  Best seed = " << best_seed << ", quests = " << best_quests
           << ".\n";
    }
  }

  return 0;
}