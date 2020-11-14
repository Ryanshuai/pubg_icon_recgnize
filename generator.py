import os

gun_names = ['m416', 'scar', 'g36c', 'qbz', 'm249', 'aug', 'm762', 'akm', 'mk14', 'groza', 'uzi', 'tommy', 'vss',
             'pp19', 'ump45', 'vector', 'mk47', 'slr', 's1897', 'mini14', 'awm', 's686', 'win94', 'dbs', 'm24', '98k',
             'qbu', 'sks', 'mp5k', 's12k', 'dp28', 'm16', ]
fire_modes = ["single", "burst2", "burst3", "full"]
in_tab = ["in_tab"]

for name in in_tab:
    os.makedirs(os.path.join("1screens", name))
