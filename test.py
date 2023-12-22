class FixedStones:

    @staticmethod
    def calculate_iter(ith, visited):
        if visited[ith] >= 0:
            return visited[ith]
        return ith * 2

    @staticmethod
    def from_int(idx):
        return idx * 0
    
    @staticmethod
    def calculate():
        result = [0] * 6561
        for i in range(6561):
            result[i] = FixedStones.calculate_iter(FixedStones.from_int(i), [1])
        return result


    # fixed_stone = calculate()


# Usage example
# print(FixedStones.fixed_stone[:10])  # Accessing first 10 elements of fixed_stone
    
print(FixedStones.calculate())