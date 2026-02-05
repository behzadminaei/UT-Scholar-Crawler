"""
Tools module - Shared utilities for data crawling and processing.

This module provides:
- API interaction functions for crawling faculty data
- File I/O utilities (JSON, directory management)
- Configuration constants
"""

import copy
import json
import math
import os
import time
from typing import Final, List, Union

import bs4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
import tqdm

# Type aliases
IntFloat = Union[int, float]
ListDict = Union[list, dict]

# Short aliases for backward compatibility (used in other scripts)
co = copy
js = json
mt = math
tm = time
tq = tqdm
req = requests
man = manifold
dec = decomposition

# Legacy type aliases (for backward compatibility)
intfloat = IntFloat
listdict = ListDict


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

SEED: Final[int] = 0
STYLE: Final[str] = 'ggplot'
SLEEP: Final[IntFloat] = 0.02

# Output paths
OUTPUT_PATH: Final[str] = 'Outputs'
OUTPUT_PATH_TEACHERS: Final[str] = f'{OUTPUT_PATH}/Teachers'
OUTPUT_PATH_NETWORKS: Final[str] = f'{OUTPUT_PATH}/Networks'
OUTPUT_PATH_PUBLICATIONS: Final[str] = f'{OUTPUT_PATH}/Publications'

# Embedding parameters
EMBEDDING_SIZE: Final[int] = 32
ITERATION_COUNT: Final[int] = 100
EPSILON: Final[float] = 0.01

# API URLs
BASE_URL: Final[str] = 'https://profile.ut.ac.ir'
EXPLORE_API_URL: Final[str] = 'https://profile.ut.ac.ir/profiles'
NETWORK_API_URL: Final[str] = 'https://profile.ut.ac.ir/profile'
PUBLICATION_API_URL: Final[str] = 'https://profile.ut.ac.ir/profile'

# API Parameters
EXPLORE_API_PARAMS: Final[dict[str, str]] = {
    'p_p_id': 'edusearch_WAR_edumanagerportlet_INSTANCE_PM4wXjldOANK',
    'p_p_lifecycle': '2',
    'p_p_state': 'normal',
    'p_p_mode': 'view',
    'p_p_cacheability': 'cacheLevelPage',
    'p_p_col_id': 'column-1',
    'p_p_col_count': '1',
    'preFilter': 'false',
    'page': '[PAGE]',
    'sortType': 'last-name',
    'currentSearchType': 'profile',
    'searchType': 'profile'
}

NETWORK_API_PARAMS: Final[dict[str, str]] = {
    'p_p_auth': '4pE0Mi3T',
    'p_p_id': 'eduteacherdisplay_WAR_edumanagerportlet',
    'p_p_lifecycle': '2',
    'p_p_state': 'normal',
    'p_p_mode': 'view',
    'p_p_cacheability': 'cacheLevelPage',
    'p_p_col_id': 'column-1',
    'p_p_col_count': '1',
    '_eduteacherdisplay_WAR_edumanagerportlet_teacherId': '[TEACHER_ID]',
    '_eduteacherdisplay_WAR_edumanagerportlet_PersonalPageScreenName': '[TEACHER_ID]',
    '_eduteacherdisplay_WAR_edumanagerportlet_teacherUserId': '[TEACHER_ID]',
    '_eduteacherdisplay_WAR_edumanagerportlet_pureSection': 'network',
    '_eduteacherdisplay_WAR_edumanagerportlet_mvcPath': '/edu-teacher-display/view_teacher.jsp',
    'cmd': 'network',
    'families': 'com.elsevier.pure.portal.ExternalPerson',  # Note: duplicate key, only last value kept
    'startYear': '2015',
    'endYear': '2025',
    'minCollaborationCount': '2'
}

PUBLICATION_API_PARAMS: Final[dict[str, str]] = {
    'p_p_auth': 'l1vQNoVb',
    'p_p_id': 'eduteacherdisplay_WAR_edumanagerportlet',
    'p_p_lifecycle': '2',
    'p_p_state': 'normal',
    'p_p_mode': 'view',
    'p_p_cacheability': 'cacheLevelPage',
    'p_p_col_id': 'column-1',
    'p_p_col_count': '1',
    '_eduteacherdisplay_WAR_edumanagerportlet_teacherId': '[TEACHER_ID]',
    '_eduteacherdisplay_WAR_edumanagerportlet_PersonalPageScreenName': '[TEACHER_ID]',
    '_eduteacherdisplay_WAR_edumanagerportlet_teacherUserId': '[TEACHER_ID]',
    '_eduteacherdisplay_WAR_edumanagerportlet_pureSection': 'publications',
    '_eduteacherdisplay_WAR_edumanagerportlet_mvcPath': '/edu-teacher-display/view_teacher.jsp',
    'limit': '25',
    'page': '1',
    'sort': 'date',
    'cmd': 'getScholarlyWorks'
}

# Initialize random seed and matplotlib style
np.random.seed(seed=SEED)
plt.style.use(style=STYLE)


# =============================================================================
# FILE I/O FUNCTIONS
# =============================================================================

def Create_Parent_Directory(file_path: str) -> None:
    """Create parent directory for a file if it doesn't exist."""
    parts = file_path.split(sep='/')
    last_part = parts[-1]
    if '.' in last_part:
        parts = parts[:-1]
    dir_path = '/'.join(parts)
    os.makedirs(name=dir_path, exist_ok=True)


def Save_JSON(data: ListDict, file_path: str) -> None:
    """Save data to JSON file."""
    Create_Parent_Directory(file_path)
    with open(file=file_path, mode='w', encoding='UTF-8') as f:
        js.dump(obj=data, fp=f, ensure_ascii=False, indent=4)


def Load_JSON(file_path: str) -> ListDict:
    """Load data from JSON file."""
    with open(file=file_path, mode='r', encoding='UTF-8') as f:
        return js.load(fp=f)


# =============================================================================
# API FUNCTIONS
# =============================================================================

def Get_Page_Teachers(page: int) -> tuple[int, list[dict]]:
    """Fetch teachers from a specific page of the API."""
    page_params = co.deepcopy(EXPLORE_API_PARAMS)
    page_params['page'] = str(page)
    response = req.get(url=EXPLORE_API_URL, params=page_params)
    content = response.content
    markup = content.decode(encoding='UTF-8')
    data = js.loads(s=markup)
    total = data['total']
    teachers = data['results']
    return (total, teachers)


def Get_Teacher_Network(teacher_id: int) -> dict:
    """Fetch network data for a specific teacher."""
    network_params = co.deepcopy(NETWORK_API_PARAMS)
    network_params['_eduteacherdisplay_WAR_edumanagerportlet_teacherId'] = str(teacher_id)
    network_params['_eduteacherdisplay_WAR_edumanagerportlet_PersonalPageScreenName'] = str(teacher_id)
    network_params['_eduteacherdisplay_WAR_edumanagerportlet_teacherUserId'] = str(teacher_id)
    response = req.get(url=NETWORK_API_URL, params=network_params)
    content = response.content
    markup = content.decode(encoding='UTF-8')
    return js.loads(s=markup)


def Get_Teacher_Publication(teacher_id: int) -> dict:
    """Fetch publication data for a specific teacher."""
    publication_params = co.deepcopy(PUBLICATION_API_PARAMS)
    publication_params['_eduteacherdisplay_WAR_edumanagerportlet_teacherId'] = str(teacher_id)
    publication_params['_eduteacherdisplay_WAR_edumanagerportlet_PersonalPageScreenName'] = str(teacher_id)
    publication_params['_eduteacherdisplay_WAR_edumanagerportlet_teacherUserId'] = str(teacher_id)
    response = req.get(url=PUBLICATION_API_URL, params=publication_params)
    content = response.content
    markup = content.decode(encoding='UTF-8')
    return js.loads(s=markup)


def Get_Teachers() -> list[dict]:
    """Fetch all teachers from the API (paginated)."""
    teachers = []
    page = 1
    max_page = mt.inf
    
    while page <= max_page:
        tm.sleep(SLEEP)
        print(f'Crawling Page {page}')
        (total, page_teachers) = Get_Page_Teachers(page)
        
        if mt.isinf(max_page):
            per_page = len(page_teachers)
            max_page = mt.ceil(total / per_page)
        
        teachers.extend(page_teachers)
        page += 1
    
    return teachers


def Get_Teacher_Networks(teachers: list[dict]) -> list[dict]:
    """Fetch networks for all teachers."""
    teacher_networks = []
    for teacher in tq.tqdm(iterable=teachers, total=len(teachers), unit='Teacher Network'):
        tm.sleep(SLEEP)
        teacher_id = int(teacher['teacherId'])
        teacher_network = Get_Teacher_Network(teacher_id)
        teacher_networks.append(teacher_network)
    return teacher_networks


def Get_Teacher_Publications(teachers: list[dict]) -> list[dict]:
    """Fetch publications for all teachers."""
    teacher_publications = []
    for teacher in tq.tqdm(iterable=teachers, total=len(teachers), unit='Teacher Publication'):
        tm.sleep(SLEEP)
        teacher_id = int(teacher['teacherId'])
        teacher_publication = Get_Teacher_Publication(teacher_id)
        teacher_publications.append(teacher_publication)
    return teacher_publications


# =============================================================================
# SAVE FUNCTIONS
# =============================================================================

def Save_Teachers(teachers: list[dict]) -> None:
    """Save teacher profiles to individual JSON files."""
    for teacher in tq.tqdm(iterable=teachers, total=len(teachers), unit='Teacher Save'):
        teacher_id = int(teacher['teacherId'])
        teacher_path = f'{OUTPUT_PATH_TEACHERS}/{teacher_id}.json'
        Save_JSON(teacher, teacher_path)


def Save_Teacher_Networks(teachers: list[dict], teacher_networks: list[dict]) -> None:
    """Save teacher networks to individual JSON files."""
    for teacher, teacher_network in tq.tqdm(
        iterable=zip(teachers, teacher_networks), 
        total=len(teachers), 
        unit='Teacher Network Save'
    ):
        teacher_id = int(teacher['teacherId'])
        teacher_network_path = f'{OUTPUT_PATH_NETWORKS}/{teacher_id}.json'
        Save_JSON(teacher_network, teacher_network_path)


def Save_Teacher_Publications(teachers: list[dict], teacher_publications: list[dict]) -> None:
    """Save teacher publications to individual JSON files."""
    for teacher, teacher_publication in tq.tqdm(
        iterable=zip(teachers, teacher_publications), 
        total=len(teachers), 
        unit='Teacher Publication Save'
    ):
        teacher_id = int(teacher['teacherId'])
        teacher_publication_path = f'{OUTPUT_PATH_PUBLICATIONS}/{teacher_id}.json'
        Save_JSON(teacher_publication, teacher_publication_path)
