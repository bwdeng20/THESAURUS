import pandas as pd
import rootutils
from dataclasses import dataclass


def insert_sub_before(original_string, insert_string, location="("):
    """
    "SimGCD()balabala" --> "SimGCD() wo SupCon balabala""
    """
    if isinstance(location, str):
        index = original_string.find(location)
    else:
        index = int(location)

    if index != -1:
        modified_string = original_string[:index] + insert_string + original_string[index:]
    else:
        modified_string = original_string
    return modified_string


def get_tabulate_report(summary_dict, interested_mts, monitor=None, prefix="repr2_test"):
    perf_tb = [["BestMonitoredCkpt"]]
    for mt in interested_mts:
        if monitor is not None:
            te_monitored = summary_dict.get(f"{prefix}/{mt}_ckpt={monitor.replace('/', '_')}", "N/A")
            perf_tb[0].append(te_monitored)
        else:
            perf_tb[0].append("N/A")
    return perf_tb


def get_tabulate_report_v2(summary_dict, interested_mts, monitor=None, prefix="repr2", stage="test"):
    perf_tb = [["BestMonitoredCkpt"], ["LastCkpt"]]
    for mt in interested_mts:
        if mt == "fit_time":
            pass
        te_last = summary_dict.get(f"{prefix}_{stage}/{mt}_ckpt=last",
                                   summary_dict.get(f"{prefix}_{stage}/{mt}", "N/A")
                                   )
        if te_last == "N/A":
            te_last = summary_dict.get(f"{prefix}_{mt}", "N/A")  # with full key name
        perf_tb[1].append(te_last)
        if monitor is not None:
            te_monitored = summary_dict.get(f"{prefix}_{stage}/{mt}_ckpt={monitor.replace('/', '_')}",
                                            summary_dict.get(f"{prefix}_{stage}/{mt}", "N/A")
                                            )
            perf_tb[0].append(te_monitored)
        else:
            perf_tb[0].append("N/A")
    return perf_tb


def class_info_from_name_space(ns):
    if ns is None:
        return None
    assert "class_path" in ns
    cls_name = ns.class_path.rsplit(".", 1)[1]
    if "init_args" not in ns:
        return cls_name()
    else:
        init_args = ns.init_args.as_dict()
        init_desc = ", ".join([f"{k}={v}" for k, v in init_args.items()])
        return f"{cls_name}({init_desc})"


def get_info_df(display_all=False):
    ProjectDir = rootutils.find_root(".", indicator=".project-root")
    info_df = pd.read_csv(ProjectDir / "data_summary.csv")
    if display_all:
        print(info_df)
    return info_df


def get_fuzzily(d, fuzzy_key: str, default=None):
    """
    d[FUZZY_KEY] --> d[fuzzy_key] --> None
    """
    return d.get(fuzzy_key, d.get(fuzzy_key.lower(), default))


if __name__ == "__main__":
    get_info_df()
