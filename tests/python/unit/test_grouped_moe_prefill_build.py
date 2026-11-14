"""Compile the grouped-prefill selector without torch, CUTLASS, or a GPU.

The CUDA 12.8 regression is in host-side template deduction. Extract the
production selector so this test exercises the same code as the wheel build.
Run directly with Python's unittest runner, or collect it with pytest.
"""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_KERNEL = _REPO / "extensions/kernel/grouped_moe_prefill.cu"
_HEADER = _REPO / "extensions/kernel/moe_grouped_gemm.h"


def _translation_unit():
    source = _KERNEL.read_text()
    # Keep the real configuration tuples, Heights, rate tables, and picker.
    configs = source.split("using moe_gemm::GroupedTiledGemm;", 1)[1]
    configs = configs.split("template <typename Configs, typename F", 1)[0]
    selector = source.split("  struct Rate {", 1)[1]
    selector = (
        "  struct Rate {"
        + selector.split("  moe_gemm::TileSplit const split[2]", 1)[0]
    )
    tile_split = re.search(
        r"struct TileSplit \{.*?\n};", _HEADER.read_text(), re.S
    ).group(0)
    return (
        """
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#define CUTLASS_HOST_DEVICE
#define TORCH_CHECK(condition, ...) \\
  do { if (!(condition)) throw std::out_of_range("config"); } while (false)
template <int M, int N, int K, int WM, int WN, int S, bool Dual>
struct G { static constexpr int kM = M; };
template <typename Big, typename Small = Big>
struct GroupedTiledGemm {
  static constexpr int kBigM = Big::kM, kSmallM = Small::kM;
  static constexpr bool kMixed = !std::is_same_v<Big, Small>;
};
namespace moe_gemm {
"""
        + tile_split
        + "\n}\n"
        + configs
        + """
int main() {
  static_assert(kNumDual == 6 && kNumDown == 8);
  for (int rows_e : {1, 16, 31, 32, 63, 64, 127, 128, 144, 250, 288,
                    1024, 8192}) {
"""
        + selector
        + """
    auto verify = [&]<typename Configs>(Rate const* rates) {
      using Hm = Heights<Configs>;
      constexpr int count = std::tuple_size_v<Configs>;
      double best = 0;
      int expected = -1;
      for (int c = 0; c < count; ++c) {
        int cfg = c;
        auto actual = pick.template operator()<Configs>(rates, cfg);
        auto reference = split_of(rates[c], Hm::big[c], Hm::small[c]);
        assert(cfg == c);
        assert(actual.big_m == reference.big_m);
        assert(actual.small_m == reference.small_m);
        assert(actual.max_small == reference.max_small);
        double value = cost(rates[c], reference);
        if (expected < 0 || value < best) {
          expected = c;
          best = value;
        }
      }
      for (int cfg : {-1, -7}) {
        auto selected = pick.template operator()<Configs>(rates, cfg);
        assert(cfg == expected);
        assert(cost(rates[cfg], selected) == best);
      }
      for (int cfg : {count, count + 1}) {
        bool rejected = false;
        try { pick.template operator()<Configs>(rates, cfg); }
        catch (std::out_of_range const&) { rejected = true; }
        assert(rejected);
      }
    };
    verify.template operator()<DualConfigs>(kDualRate);
    verify.template operator()<DownConfigs>(kDownRate);
  }
}
"""
    )


class GroupedPrefillBuildTest(unittest.TestCase):
    def test_no_dependent_array_size_deduction(self):
        # Host compilers accept this expression, but nvcc 12.8 does not.
        self.assertNotIn("std::size(Hm::big)", _KERNEL.read_text())

    def _compile_and_run(self, compiler, flags):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "selector.cu"
            binary = Path(directory) / "selector"
            source.write_text(_translation_unit())
            result = subprocess.run(
                [compiler, *flags, str(source), "-o", str(binary)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                result.returncode, 0, result.stdout + result.stderr
            )
            subprocess.run([str(binary)], check=True, timeout=30)

    @unittest.skipUnless(shutil.which("g++"), "g++ is not installed")
    def test_host_compiler(self):
        self._compile_and_run(
            "g++", ["-x", "c++", "-std=c++20", "-Wall", "-Wextra", "-Werror"]
        )

    @unittest.skipUnless(shutil.which("nvcc"), "CUDA compiler is not installed")
    def test_cuda_compiler(self):
        self._compile_and_run("nvcc", ["-std=c++20"])


if __name__ == "__main__":
    unittest.main()
