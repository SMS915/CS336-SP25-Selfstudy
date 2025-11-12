from typing import Dict, List, Set, Hashable, Iterable

class UnionFind:
    """
    一个高效的、基于字典实现的并查集（Disjoint Set Union）数据结构。

    这个数据结构用于追踪一组元素被分割为多个不相交的子集的情况，
    并能高效地合并这些子集。
    """

    def __init__(self, elements: Iterable[Hashable]):
        """
        初始化并查集。
        初始时，每个元素都自成一个集合。

        Args:
            elements (Iterable[Hashable]): 需要处理的所有元素的集合或列表，
                                           例如所有的文档ID [0, 1, 2, ...]。
        """
        # `parent` 字典用于存储每个元素的父节点。
        # 如果 parent[x] == x，则 x 是其所在集合的根节点。
        self.parent: Dict[Hashable, Hashable] = {elem: elem for elem in elements}
        
        # `rank` 字典用于路径压缩优化，记录每个根节点树的高度。
        # 这有助于保持树的扁平，加快查找速度。
        self.rank: Dict[Hashable, int] = {elem: 0 for elem in elements}

    def find(self, item: Hashable) -> Hashable:
        """
        查找元素所属集合的根节点（代表元素）。
        同时进行路径压缩优化。

        Args:
            item (Hashable): 要查找的元素。

        Returns:
            Hashable: 该元素所属集合的根节点。
        """
        # 如果一个元素的父节点不是它自己，说明它不是根节点
        if self.parent[item] != item:
            # 递归地向上查找，直到找到根节点
            # 【路径压缩优化】: 将查找路径上的所有节点的父节点直接指向根节点
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: Hashable, item2: Hashable) -> None:
        """
        合并两个元素所在的集合。

        Args:
            item1 (Hashable): 第一个元素。
            item2 (Hashable): 第二个元素。
        """
        # 找到两个元素各自的根节点
        root1 = self.find(item1)
        root2 = self.find(item2)

        # 如果它们已经在同一个集合中（根节点相同），则无需任何操作
        if root1 == root2:
            return

        # 【按秩合并优化】: 将较矮的树合并到较高的树上，避免树变得过高
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            # 如果两棵树高度相同，任意合并，并将新根节点的高度加一
            self.parent[root2] = root1
            self.rank[root1] += 1

    def get_clusters(self) -> List[Set[Hashable]]:
        """
        从并查集结构中提取出所有的聚类（集合）。

        Returns:
            List[Set[Hashable]]: 一个列表，其中每个元素是一个代表聚类的集合。
        """
        clusters: Dict[Hashable, Set[Hashable]] = {}
        for item in self.parent:
            # 找到每个元素的根节点
            root = self.find(item)
            # 将该元素添加到以其根节点为键的集合中
            if root not in clusters:
                clusters[root] = set()
            clusters[root].add(item)
        
        return list(clusters.values())