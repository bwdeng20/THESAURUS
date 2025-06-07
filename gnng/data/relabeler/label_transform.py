from gnng.data.relabeler.base import LabelTransform
from gnng.typing import PyGData
from gnng.data.relabeler.functional import target_distribution_from_nbr_dict, target_distribution_from_adjacency


class SemanticGraphLabel(LabelTransform):
    def __init__(self, label_nbr_dict, pivot_p=0.9, undirected_relation=True, copy=True, within_old=None):
        super().__init__(copy)
        self.label_nbr_dict = label_nbr_dict
        self.pivot_p = pivot_p
        self.undirected_relation = undirected_relation
        self.within_old = within_old

    def transform_label(self, g: PyGData) -> PyGData:
        label = g.get("y", g.get("label", None))
        g["y"] = target_distribution_from_nbr_dict(
            label, self.label_nbr_dict, self.pivot_p, undirected=self.undirected_relation, within_old=self.within_old
        )
        return g


class SemanticAdjLabel(LabelTransform):
    def __init__(self, Ay, undirected_relation=True, copy=True, within_old=None):
        super().__init__(copy)
        self.Ay = Ay
        self.undirected_relation = undirected_relation
        self.within_old = within_old

    def transform_label(self, g: PyGData) -> PyGData:
        label = g.get("y", g.get("label", None))
        g["y"] = target_distribution_from_adjacency(
            label, self.Ay, self.undirected_relation, within_old=self.within_old
        )
        return g
