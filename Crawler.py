from Tools import *

Teachers = Get_Teachers()

Teacher_Networks = Get_Teacher_Networks(Teachers)

Teacher_Publications = Get_Teacher_Publications(Teachers)

Save_Teachers(Teachers)

Save_Teacher_Networks(Teachers, Teacher_Networks)

Save_Teacher_Publications(Teachers, Teacher_Publications)
