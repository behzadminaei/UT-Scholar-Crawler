from Tools import *

Teacher_IDs = list(map(lambda x: int(x[:-5]), os.listdir(path=OUTPUT_PATH_TEACHERS)))

Aggregated_Profiles = []

Aggregated_Nodes = {}

Aggregated_Edges = {}

for Teacher_ID in Teacher_IDs:
    
    Teacher_Data = Load_JSON(f'{OUTPUT_PATH_TEACHERS}/{Teacher_ID}.json')
    Teacher_Network_Data = Load_JSON(f'{OUTPUT_PATH_NETWORKS}/{Teacher_ID}.json')
    Teacher_Publication_Data = Load_JSON(f'{OUTPUT_PATH_PUBLICATIONS}/{Teacher_ID}.json')
    
    Aggregated_Profile = {}
    Aggregated_Profile['ID'] = Teacher_ID
    Aggregated_Profile['First_Name_FA'] = Teacher_Data['firstName']
    Aggregated_Profile['Last_Name_FA'] = Teacher_Data['lastName']
    Aggregated_Profile['First_Name_EN'] = Teacher_Data['firstName_en_US']
    Aggregated_Profile['Last_Name_EN'] = Teacher_Data['lastName_en_US']
    Aggregated_Profile['Degree'] = Teacher_Data['degree']
    Aggregated_Profile['Email'] = Teacher_Data['email']
    if 'organistaions' in Teacher_Data:
        Aggregated_Profile['Faculty'] = Teacher_Data['organistaions'][0]['name']
    else:
        Aggregated_Profile['Faculty'] = None
    
    Year_Languages = [[i['group'], i['lang']] for i in Teacher_Publication_Data['results']]

    Year_Language_Grouped_Count = {'Persian': {},
                                   'Foreign': {}}

    if len(Year_Languages) > 0:
        Min_Year = min([Year_Language[0] for Year_Language in Year_Languages])
        Max_Year = max([Year_Language[0] for Year_Language in Year_Languages])
        Year_Range = list(range(Min_Year, Max_Year + 1))
        Year_Language_Grouped_Count['Persian'] = {Year: 0 for Year in Year_Range}
        Year_Language_Grouped_Count['Foreign'] = {Year: 0 for Year in Year_Range}
        for Year_Language in Year_Languages:
            Year = Year_Language[0]
            Language = 'Persian' if Year_Language[1] == 'persian' else 'Foreign'
            Year_Language_Grouped_Count[Language][Year] += 1
    
    Aggregated_Profile['Publication'] = Year_Language_Grouped_Count

    Aggregated_Profiles.append(Aggregated_Profile)

    if len(Teacher_Network_Data['edges']) > 0:
        for Node in Teacher_Network_Data['nodes']:
            Node_ID = int(Node['id'])
            if Node_ID not in Aggregated_Nodes:
                Cleaned_Node = {}
                Cleaned_Node['Name'] = Node['properties']['name']
                Cleaned_Node['Family'] = Node['properties']['family'].split(sep='.')[-1]
                Aggregated_Nodes[Node_ID] = Cleaned_Node
        
        for Edge in Teacher_Network_Data['edges']:
            Node_1, Node_2 = sorted([int(Edge['from']['id']), int(Edge['to']['id'])])
            if Node_1 not in Aggregated_Edges:
                Aggregated_Edges[Node_1] = []
            if Node_2 not in Aggregated_Edges[Node_1]:
                Aggregated_Edges[Node_1].append(Node_2)

Save_JSON(Aggregated_Profiles, f'{OUTPUT_PATH}/Aggregated_Profiles.json')
Save_JSON(Aggregated_Nodes, f'{OUTPUT_PATH}/Aggregated_Nodes.json')
Save_JSON(Aggregated_Edges, f'{OUTPUT_PATH}/Aggregated_Edges.json')