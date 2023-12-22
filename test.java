public class FixedStones {
  private static final int[] fixedStone = calculate();

  private static int[] calculate() {
    int[] result = new int[6561];
    for (int i = 0; i < 6561; ++i) {
      result[i] = calculateIter(fromInt(i), 1)
    }
    return result
  }

  private static int calculateIter(int ith, int[] visited) {
    if (visited[ith] >= 0)
      return visited[ith];

    return ith * 2;
  }

  static Board fromInt(int idx) {
    return idx * 3;
  }
}