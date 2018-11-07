import json


def add_nodes(out_target, data, offset, type, items, node_pos, type_num=None):
    height = 120
    width = 100
    num = max([len(d) for d in data])
    agent_type = type + "_" + str(type_num) if type_num is not None else type
    out_target.append({"data": {"id": type + "_start"},
                       "position": {"x": float((num - 1) / 2) * width + offset,
                                    "y": height * len(data) + 40},
                       "classes": agent_type})
    node_pos[out_target[-1]["data"]["id"]] = (out_target[-1]["position"]["x"],
                                             out_target[-1]["position"]["y"])
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
            node_pos[n] = (node["position"]["x"], node["position"]["y"])


def add_edges(out_target, data, type, node_pos, nodes):
    for s, gs in data.items():
        if s is None:
            s = type + "_start"
        for g, kind in gs.items():
            edge = {}
            edge["data"] = {"id": s + "_" + g, "source": s, "target": g}

            # print(node_pos)
            # print(s, g)
            if kind != 0:
                node = {}
                node["data"] = {"id": s + "_" + g + "_cost"}
                node["position"] = {"x": (node_pos[s][0] + node_pos[g][0]) / 2,
                                    "y": (node_pos[s][1] + node_pos[g][1]) / 2}
                node["classes"] = "cost_" + str(kind)
                nodes.insert(0, node)
                # edge["classes"] = "edge_" + str(kind)
            out_target.append(edge)


def add_edges_model(out_target, data, type):
    for s, gs in data.items():
        if s is None:
            s = type + "_start"
        out_target[s] = list(gs.keys())


def make_json(data, algo, obj):
    json_data = {}
    json_data["nodes"] = []
    num = max([len(d) for d in data.h_node])
    node_pos = {}
    add_nodes(json_data["nodes"], data.h_node, 50, "human", data.items, node_pos)
    add_nodes(json_data["nodes"], data.r_node, num * 120 + 50, "agent", data.items, node_pos,
              algo + 1)
    json_data["edges"] = []
    add_edges(json_data["edges"], data.h_edge, "human", node_pos, json_data["nodes"])
    add_edges(json_data["edges"], data.r_edge, "agent", node_pos, json_data["nodes"])
    json_data["model_h_edge"] = {}
    json_data["model_r_edge"] = {}
    add_edges_model(json_data["model_h_edge"], data.h_edge, "human")
    add_edges_model(json_data["model_r_edge"], data.r_edge, "agent")
    json_data["height"] = len(data.h_node) * 120 + 90
    json_data["target"] = obj
    return json_data
