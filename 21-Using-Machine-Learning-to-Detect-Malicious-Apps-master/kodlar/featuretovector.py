#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Zhenghao Lu'


import androguard.core.api_specific_resources as apis
import pickle
import re


# put all permissions and apis in a list and a dict
permissions = apis.load_permissions(25)
mappings = apis.load_permission_mappings(25)

list_perm_api = []
dict_perm_api = {}
for permission in permissions.keys():
    list_perm_api.append(permission)
for api in mappings.keys():
    m = re.match(r'(.*\;?)\-(.*?)\-.*', api)
    the_api = m.group(1) + m.group(2)
    list_perm_api.append(the_api)
for index, perm_api in enumerate(list_perm_api):
    dict_perm_api[perm_api] = index
with open('list_perm_api.p', 'wb') as fp:
    pickle.dump(list_perm_api, fp)
with open('dict_perm_api.p', 'wb') as fp:
    pickle.dump(dict_perm_api, fp)