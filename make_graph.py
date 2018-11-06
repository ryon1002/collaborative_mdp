import json

def add_nodes(out_target, data, offset, type, items, type_num=None):
    height = 120
    width = 100
    num = max([len(d) for d in data])
    agent_type = type + "_" + str(type_num) if type_num is not None else type
    out_target.append({"data": {"id": type + "_start"},
                       "position": {"x": float((num - 1) / 2) * width + offset,
                                    "y": height * len(data) + 40},
                       "classes": agent_type})
    for i, nodes in enumerate(data):
        l_offset = float((num - len(nodes))) / 2
        for j, n in enumerate(nodes):
            node = {}
            node["data"] = {"id": n}
            node["position"] = {"x": width * (j + l_offset) + offset,
                                "y": (len(data) - 1 - i) * height + 50}
            for k, g in enumerate(items):
                if n in g:
                    node["classes"] = "goal_" + str(k)
            out_target.append(node)


def add_edges(out_target, data, type):
    for s, gs in data.items():
        if s is None:
            s = type + "_start"
        for g, kind in gs.items():
            edge = {}
            edge["data"] = {"id": s + "_" + g, "source": s, "target": g}
            if kind != 0:
                edge["classes"] = "edge_" + str(kind)
            out_target.append(edge)

def add_edges_model(out_target, data, type):
    for s, gs in data.items():
        if s is None:
            s = type + "_start"
        out_target[s] = list(gs.keys())

def make_json(data, algo, obj):
    json_data = {}
    json_data["nodes"] = []
    add_nodes(json_data["nodes"], data.h_node, 50, "human", data.items)
    add_nodes(json_data["nodes"], data.r_node, 530, "agent", data.items, algo + 1)
    json_data["edges"] = []
    add_edges(json_data["edges"], data.h_edge, "human")
    add_edges(json_data["edges"], data.r_edge, "agent")
    json_data["model_h_edge"] = {}
    json_data["model_r_edge"] = {}
    add_edges_model(json_data["model_h_edge"], data.h_edge, "human")
    add_edges_model(json_data["model_r_edge"], data.r_edge, "agent")
    json_data["height"] = len(data.h_node) * 120 + 90
    json_data["target"] = obj
    return json_data
