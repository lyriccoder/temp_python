import pandas as pd
import subprocess
import os
import re
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import TimeoutError
import argparse
import os
import multiprocessing
import traceback
from cchardet import detect
import sys
from multiprocessing import Pool
from program_slicing.graph.parse import tree_sitter_ast_java
from program_slicing.graph.parse import tree_sitter_parsers
from program_slicing.graph.cdg import ControlDependenceGraph
from program_slicing.graph.statement import Statement, StatementType
from program_slicing.graph.point import Point
import csv   
from collections import defaultdict, OrderedDict
from program_graphs.ddg.ddg import mk_ddg
from program_graphs.cfg.parser.java.parser import mk_cfg
from program_graphs.cfg import CFG
from program_graphs.adg import parse_java
from typing import List, Tuple, Set, Optional
import networkx as nx
import numpy as np

CycloComplexity = int

    
sys.setrecursionlimit(100000)

def __traverse(root):
    yield root
    if root.children:
        for child in root.children:
            for result in __traverse(child):
                yield result

def detect_encoding_of_file(filename: str):
    with open(filename, 'rb') as target_file:
        return detect_encoding_of_data(target_file.read())


def detect_encoding_of_data(data: bytes):
    return detect(data)['encoding']

 
def read_text_with_autodetected_encoding(filename: str):
    with open(filename, 'rb') as target_file:
        data = target_file.read()

    if not data:
        return ''  # In case of empty file, return empty string

    encoding = detect_encoding_of_data(data) or 'utf-8'
    return data.decode(encoding)


def cc(source_code: str) -> CycloComplexity:
    adg = parse_java(source_code)
    #cfg = parse_cfg(source_code)
    cfg = adg.to_cfg()
    E = len(cfg.edges())
    N = len(cfg.nodes())
    P = nx.number_weakly_connected_components(cfg)
    return E - N + 2 * P


def get_json_with_tokens(groupby_objects):
    filepath, group_item, dataset_dir = groupby_objects
    total_test_list = []
    ignored_file = None
    #print('begin ', filepath)
    try:
        for _, item in group_item.iterrows():
            #print(f'{_}, item {item}')
            data_dir = item.data_dir
            fullpath =  Path(dataset_dir) / data_dir.replace('/', '', 1) / str(filepath)
            #print(f'fullpath {fullpath} exists? {fullpath.exists()}')
            #fullpath =  Path(dataset_dir) / str(filepath)
            source_code = read_text_with_autodetected_encoding(str(fullpath))
            bytes_size = fullpath.stat().st_size
            mb_size = bytes_size >> 20
            file_dict = {'filename': fullpath.resolve(), 'bytes': bytes_size, 'kb': bytes_size >> 10, 'mb': mb_size}
            if mb_size > 0:
                ignored_file = fullpath
                total_test_list.append(file_dict)
                return total_test_list, ignored_file
            source_code_bytes = bytes(source_code, "utf8")
            ast = tree_sitter_ast_java.parse(source_code).root_node
            methods = {}
            comments_dicts = []
            for node in __traverse(ast):
                if node.type == 'method_declaration':
                    method_name_in_file = source_code_bytes[node.child_by_field_name('name').start_byte:node.child_by_field_name('name').end_byte].decode('utf-8')
                    methods[tuple([method_name_in_file, node.start_point[0]])] = node
                    #print(fullpath, method_name_in_file, node.start_point[0])
                if node.type in {"line_comment", "block_comment", "comment"}:
                    comment = source_code_bytes[node.start_byte:node.end_byte].decode('utf-8')
                    if comment.lower().find('generated') > -1:
                        #print(f'class {fullpath}; comment {comment}')
                        comments_dicts.append({'line': node.start_point[0]})
            method_name_in_config = item['methodname']
            method_startline = item['startline']
            found_method = methods.get((method_name_in_config, method_startline))
            if found_method:
                #print(f'Found {method_name_in_file}')
                method_code = source_code_bytes[found_method.start_byte:found_method.end_byte].decode('utf-8')
                #print(f'{method_code}')
                cyclo_complexity = cc(method_code)
                counts = defaultdict(int)
                for x in ['for_statement', 'decimal_integer_literal', 'character_literal', 
                    'generated_comment', 'while_statement', 'if_statement', 'variable_declarator', 'lines_n', 
                    'local_variable_declaration', 'switch_expression', 'case']:
                        counts[x] = 0
                counts['cyclo_complexity'] = cyclo_complexity 
                counts['method_name'] = method_name_in_config
                for node in __traverse(found_method):
                    #print(node)
                    #if node.type == 'method_declaration':
                        #method_name_in_file = source_code_bytes[node.start_byte:node.end_byte].decode('utf-8')
                        #print(method_name_in_file)
                        #cyclo_complexity = cc(method_name_in_file)
                        #ccs.append(cyclo_complexity)
                        #methods.append((method_name_in_file, node.start_point[0], node))
                    
                    if node.type in {"if_statement"}:
                        counts[node.type] += 1
                    if node.type in {"while_statement"}:
                        counts[node.type] += 1
                    if node.type in {"for_statement"}:
                        counts[node.type] += 1
                    if node.type in {"local_variable_declaration"}:
                        counts[node.type] += 1
                    if node.type in {"variable_declarator"}:
                        counts[node.type] += 1
                    if node.type in {"method_invocations"}:
                        counts[node.type] += 1
                    if node.type in {"character_literal"}:
                        counts[node.type] += 1
                    if node.type in {"switch_expression"}:
                        counts[node.type] += 1   
                    if node.type in {"case"}:
                        counts[node.type] += 1    
                    if node.type in {"decimal_integer_literal"}:
                        #qq = source_code_bytes[node.child_by_field_name('name').start_byte:node.child_by_field_name('name').end_byte].decode('utf-8')
                        #print(qq)
                        counts[node.type] += 1
                d = {**counts, **{'generated_comment': comments_dicts}, **file_dict}
                #d['cyclo_complexity'] = np.mean(ccs)
                d['lines_n'] = len([x for x in source_code.split('\n') if x.strip()])
                total_test_list.append(d)
                #else:
                    #print(f'Method {method_name_in_config} not found in {fullpath}')

    #except FileNotFoundError as e:
        #print(f'Cannot find file {fullpath}')
        #pass
    except Exception as e:
        print(traceback.format_exc())
        print(f'Cannot read/open/parse file {fullpath} {str(e)}')
    
    #print('end ', filepath)
    return total_test_list, ignored_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--dir', '-d', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input, sep=',', encoding='utf-8')
    before_size = df.shape[0]
    print(f'df before {before_size}')
    df = df[~df['filepath'].str.contains('zxiaofan__JDK')]
    perc = (df.shape[0]/float(before_size)) * 100
    print(f'df after {df.shape[0]}, remained {perc}% items')
    cpu_count = multiprocessing.cpu_count()
    #cpu_count = 1
    count = 0
    manager = multiprocessing.Manager()
    lst = manager.list()
    groupby_objects = [(filepath, group_item, args.dir) for filepath, group_item in df.groupby('filepath')]
    files = []
    print(f'CPU : {cpu_count}')
    fields=['filename','lines_n', 'for_statement', 'decimal_integer_literal', 'character_literal', 'generated_comment', 'while_statement', 'if_statement',
    'variable_declarator', 'cyclo_complexity', 'lines', 'local_variable_declaration', 'switch_expression', 'case', 'kb', 'bytes', 'mb', 'method_name']
    if Path(args.output).exists():
        Path(args.output).unlink()
    if Path('ignored_files.csv').exists():
        Path('ignored_files.csv').unlink()
    rows_n = 0
    pool = Pool(cpu_count)
    with open('ignored_files.csv', "a", newline='\n') as output_ignored:
        writer_ignored = csv.DictWriter(output_ignored, fieldnames=['filename', 'bytes', 'kb', 'mb'])
        writer_ignored.writeheader()
        with open(args.output, "a", newline='\n') as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            res_p = list(tqdm(pool.imap(get_json_with_tokens, groupby_objects), total=len(groupby_objects)))
            if res_p:
                for j in res_p:
                    file_ignored = j[-1]
                    results = j[0]
                    
                    if file_ignored is None:
                        #print('file_ignored ', file_ignored)
                        try:
                            if results:
                                #count += 1
                                for j in results:
                                    rows_n += 1
                                    #print('####################################\n', j, '\n###################################')
                                    lst.append(j)
                                if (rows_n % 500) == 0:
                                    sys.stdout.flush()
                                    for x in lst:
                                        writer.writerow(x)
                                    output.flush()
                                    lst[:] = []
                        except Exception as e:
                            print(f'{str(e)}')
                    else:
                        #print('file_ignored2 ', results)
                        print('file_ignored ', file_ignored)
                        for m in results:
                            #print('file_ignored ', m)
                            writer_ignored.writerow(m)  
                            output_ignored.flush()
                for x in lst:
                    writer.writerow(x)
            print(f'Finished, items {rows_n}')
