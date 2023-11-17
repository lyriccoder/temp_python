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
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


CycloComplexity = int

sys.setrecursionlimit(100000)

m_code = '''
	protected Expression transformClosureExpression ( ClosureExpression ce) 
    {
        boolean oldInClosure = inClosure;
        inClosure = true;
        Parameter[] paras = ce.getParameters();
        if (paras != null) {
            for (Parameter para : paras) {
                ClassNode t = para.getType();
                resolveOrFail(t, ce);
                visitAnnotations(para);
                if (para.hasInitialExpression()) {
                    Object initialVal = para.getInitialExpression();
                    if (initialVal instanceof Expression) {
                        para.setInitialExpression(transform((Expression) initialVal));
                    }
                }
                visitAnnotations(para);
            }
        }
        Statement code = ce.getCode();
        if (code != null) code.visit(this);
        inClosure = oldInClosure;
        return ce;
    }
'''
m_code = '''    private void writeObject(java.io.ObjectOutputStream out) throws java.io.IOException {
      try {
        int i = 0;
      }
      finally {
            switch (te.getMessage()) {
                case "Unable to commit: transaction marked for rollback":
                    // don't log as error, this happens if there's a ConcurrentUpdateException
                    // at transaction end inside VCS
                    isRollbackDuringCommit = true;
                    // $FALL-THROUGH$
                case "Unable to commit: Transaction timeout":
                    // don't log either
                    log.debug(msg, e);
                    break;
                default:
                    log.error(msg, e);
            }
            throw new TransactionRuntimeException(e.getMessage(), e);
      }
      
    }
'''
#with open('/hdd/emaslov/pmd-bin-6.46.0/large_disk/method_name_data/code2vec/java-large/training/apache__hive/standalone-metastore/src/gen/thrift/gen-javabean/org/apache/hadoop/hive/metastore/api/ThriftHiveMetastore.java') as f:
    #m_code = f.read()
adg = parse_java(m_code)
ast = adg.to_ast()
nodes = [ast.nodes[x].get('ast_node') for x in ast if ast.nodes[x].get('ast_node')]
#print(nodes)
print([x for x in nodes if x.type == 'throw_statement'])
ddg = adg.to_ddg()

G=nx.Graph()
egde_labels = {}
# Add nodes and edges
for x, y in ddg.edges:
    #print(name, type(name))
    x_node = ast.nodes[x].get('ast_node')
    y_node = ast.nodes[y].get('ast_node')
    #print(node, type(node))
    x_label = ddg._node_to_label(x)
    y_label = ddg._node_to_label(y)
    egde_labels[x] = x_label + f'; start_pos={x_node.start_point}'
    egde_labels[y] = y_label + f'; start_pos={y_node.start_point}'
    #print(x_label, y_label, type(x_label), type(y_label))
    G.add_edge(x, y)
 

print(G) 
plt.figure(figsize=(20,14))
# <matplotlib.figure.Figure object at 0x7f1b65ea5e80>
pos = nx.nx_pydot.graphviz_layout(G)
nx.draw(G, pos = pos, \
    node_size=1200, node_color='lightblue', linewidths=0.25, \
    font_size=10, font_weight='bold', with_labels=False)
nx.draw_networkx_labels(G, pos, egde_labels, font_size=16)
plt.savefig('labels.png')
#plt.show()    ## plot2.png attached
#nx.draw(G, pos=graphviz_layout(G), with_labels = True)
comp = list(nx.connected_components(G))

#pdot = nx.nx_pydot.to_pydot(G)
#print(pdot)
#print(ddg.edges)
#print(dir(ddg))
#print(nx.connected_components(ddg.to_undirected()))
#nx.nx_pydot.to_pydot(ddg.to_undirected())
#print(ddg)
