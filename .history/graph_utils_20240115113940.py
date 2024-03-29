import networkx as nx
import re

def create_graph(self):
        all_sections = self.all_sections
        
        max_string_length = 50
    
        parent_list, parent_index_list = [], []
       
        for i, section_segment_list in enumerate(all_sections):
            section = section_segment_list[0].sec
            section_id = i
            section_name = section.psection()['name']
            match = re.search(r'\.(.*?)\[', section_name)
            section_type = match.group(1)
            L = section.psection()['morphology']['L']

            parent_list.append(section_name)
            parent_index_list.append(section_id)

            if i == 0:
                parent_name = 'None'
                parent_id = 0
                
            else:
                parent = section.psection()['morphology']['parent'].sec
                parent_name = parent.psection()['name']
                parent_id = parent_index_list[parent_list.index(parent_name)]
            
            # create data
            data_to_append = {'parent_id': parent_id,
                    'section_id': section_id,
                    'parent_name': parent_name,
                    'section_name': section_name,
                    'length': L,
                    'section_type': section_type}

            # self.section_df = self.section_df.append(data_to_append, ignore_index=True)
            self.section_df = pd.concat([self.section_df, pd.DataFrame(data_to_append, index=[0])], ignore_index=True)
            
        self.section_df.to_csv("cell1.csv", encoding='utf-8', index=False)
        Data = open('cell1.csv', "r")
        next(Data, None)  # skip the first line in the input file

        Graphtype = nx.Graph()
        DiGraphtype = nx.DiGraph()
        self.G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                                  nodetype=int, data=(('parent_name', str), ('section_name', str),
                                                      ('length', float), ('section_type', str)))
        Data = open('cell1.csv', "r")
        next(Data, None)  # skip the first line in the input file
        self.DiG = nx.parse_edgelist(Data, delimiter=',', create_using=DiGraphtype,
                                     nodetype=int, data=(('parent_name', str), ('section_name', str),
                                                      ('length', float), ('section_type', str)))
        self.sp = dict(nx.all_pairs_shortest_path(self.G))

def _set_graph_order(self):
    order_dict = nx.single_source_shortest_path_length(self.G, 0)

    # 创建一个空字典来保存分类结果
    self.class_dict = {}

    # 将每个点根据距离分类
    for node, order in order_dict.items():
        if order not in self.class_dict:
            self.class_dict[order] = []
        self.class_dict[order].append(node)

    # 获取最远的点到soma的距离（k值）
    max_order = max(order_dict.values())

    # 输出分类结果
    # for i in range(max_order + 1):
    #     print(f"Class {i}: {self.class_dict.get(i, [])}")
