from pathlib import Path
from typing import Dict, Any, List, Tuple
import os
import pandas as pd
import requests
from random import choice
from tqdm import tqdm
import json
from pathlib import Path
import os
import pandas as pd
import requests
from random import choice
from tqdm import tqdm
import time
import asyncio
import aiohttp
import json
import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from concurrent.futures import TimeoutError
from csv import DictWriter, QUOTE_MINIMAL
from functools import partial
from os import cpu_count, getenv, makedirs, sched_getaffinity
from typing import List, Dict
import uuid
from io import StringIO, BytesIO
from lxml import html, etree
import traceback

from typing import List, Tuple, Set, Dict
from loguru import logger
from tree_sitter.binding import Node
from program_graphs.adg.parser.java.utils import parse_ast_tree_sitter
from urllib.parse import unquote
import sys
import requests
requests.urllib3.disable_warnings()


# HERE there must be tokens for github API
tokens = [
]

logger.remove(0)
if not Path('logs').exists():
    Path('logs').absolute().mkdir()

logger.add(
    "logs/debug.log", rotation="500 MB", filter=lambda record: record["level"].name == "DEBUG", backtrace=True,
    diagnose=True)
logger.add(
    "logs/info.log", rotation="500 MB", filter=lambda record: record["level"].name == "INFO", backtrace=True,
    diagnose=True)
logger.add(
    "logs/error.log", rotation="500 MB", filter=lambda record: record["level"].name == "ERROR", backtrace=True,
    diagnose=True)


def get_random_token(token_dict) -> Tuple[str, str]:
    return choice(token_dict)


def traverse(root) -> Node:
    yield root
    if root.children:
        for child in root.children:
            for result in traverse(child):
                yield result


def get_tree_sitter_node_name(node: Node, code: str) -> str:
    if node.type == "field_declaration":
        var_decl_node = \
            [x for x in node.children if x.type == "variable_declarator"][0]
        return get_tree_sitter_node_name(var_decl_node, code)

    for n in node.children:
        if n.type == "identifier":
            name = \
                bytes(code, "utf-8")[n.start_byte:n.end_byte].decode("utf-8")
            return name


async def process_commit(
        url: str,
        cwe_id: str,
        cve_id: str,
        session: Any,
        args: Namespace) -> None:
    global tokens
    try:
        headers = await get_random_auth_headers(tokens)
        logger.debug(f'Connecting to {url} {type(session)}')
        repo_name = url.split('/commits')[0].split('repos/')[1]
        #logger.debug(f'Repo name: {repo_name}')
        async with session.get(url, headers=headers, ssl=False, raise_for_status=True) as response:
            logger.debug(f'Resp: {response}')
            res = await response.read()
            content = json.loads(res.decode())
            changed_files = content.get('files')
            #logger.debug(f'changed_files: {changed_files}')
            commit_sha = content.get('sha')
            #logger.debug(f'After reponame {changed_files}')
            if changed_files:
                for i, file in enumerate(changed_files, start=1):
                    logger.debug('Iter before start')
                    await find_changed_func_in_file(
                        args,
                        commit_sha,
                        cwe_id,
                        cve_id,
                        file,
                        url,
                        repo_name,
                        i
                    )
    except asyncio.CancelledError:
        raise
    except:
        logger.error(f'Error in {url}, {class_name}, {func_name}')
        traceback.print_exc()


async def find_changed_func_in_file(
        args: Namespace,
        commit_sha: str,
        cwe_id: str,
        cve_id: str,
        file: Dict[str, Any],
        url: str,
        repo_name: str,
        iter: int) -> None:
    global tokens
    full_file = Path(file['filename'])
    #filename = full_file.stem
    raw_url = file['raw_url']
    repo_url = url.split('/commits')[0].replace('repos/', '')

    if str(full_file).lower().endswith(".C") or \
            str(full_file).lower().endswith(".cc") or \
            str(full_file).lower().endswith(".c") or \
            str(full_file).lower().endswith(".cpp") or \
            str(full_file).lower().endswith(".cxx") or \
            str(full_file).lower().endswith(".cppm") or \
            str(full_file).lower().endswith(".ixx") or \
            str(full_file).lower().endswith(".cp") or \
            str(full_file).lower().endswith(".c++"):

        headers = await get_random_auth_headers(tokens)
        # await asyncio.sleep(1)
        logger.debug('Before commit request')
        async with aiohttp.ClientSession() as session3:
            async with session3.get(raw_url, headers=headers, ssl=False, raise_for_status=True) as response1:
                after_file_code_bytes = await response1.read()
                logger.debug('After request1')
                extention = full_file.suffixes[0]
                await save_to_file(
                    cwe_id, commit_sha, args.output, False, iter, cve_id, repo_name, after_file_code_bytes, extention)
                logger.debug('After save_file_after')

                # check history and get prev version
                url_for_certain_file_by_certain_sha = f'{repo_url}/commits?sha={commit_sha}&path={file["filename"]}'.replace(
                    'api.github.com', 'api.github.com/repos')
                # await asyncio.sleep(1)
                #logger.debug(f'connecting to {url_for_certain_file_by_certain_sha}')
                #logger.debug(f'Next token {headers}')
        headers = await get_random_auth_headers(tokens)
        prev_commits = requests.get(url_for_certain_file_by_certain_sha, headers=headers, verify=False).json()
        #print(res.status_code)
        #logger.debug(res.content)
        #async with aiohttp.ClientSession() as session4:
        #   async with session4.get(url_for_certain_file_by_certain_sha, headers=headers, ssl=False, raise_for_status=True) as response4:
        #         #logger.debug('RESP4', response4)
        #         #logger.debug(f'got prev resp {url_for_certain_file_by_certain_sha}')
                #res4 = await response4.read()
                #logger.debug('After request2')
                #prev_commits = res4.decode("utf-8")
        logger.debug('After getting prev_commits')
        # we need previous commit, so get commit after the first,
        # since the first is the current commit
        if len(prev_commits) > 1:
            old_version_commit = prev_commits[1].get('sha')
            old_commit_url_for_file = f'{repo_url}/commits/{old_version_commit}/{file["filename"]}'.replace(
                'api.github.com', 'github.com').replace('commits/', 'raw/')
            #logger.debug(f'Connecting to old_commit_url_for_file {old_commit_url_for_file}')
            headers = await get_random_auth_headers(tokens)
            # await asyncio.sleep(1)
            prev_file_code_bytes = requests.get(old_commit_url_for_file, headers=headers, verify=False).content
            # async with session4.get(
            #         old_commit_url_for_file,
            #         headers=headers,
            #         ssl=False,
            #         raise_for_status=True) as response3:
            #prev_file_code_bytes = await response3.read()
            logger.debug('After request resp3')
            await save_to_file(
                cwe_id,
                commit_sha,
                args.output,
                True,
                iter,
                cve_id,
                repo_name,
                prev_file_code_bytes,
                extention)
            logger.debug('After save before file')
        else:
            logger.debug(f'Can\'t find prev commit for {filename} {class_name}')


async def save_to_file(
        cwe_id: str,
        commit_sha: str,
        output_folder: str,
        is_vul: bool,
        num: int,
        cve_id: str,
        repo_name: str,
        binary_text: bytes,
        extention: str) -> None:
    logger.debug('inside save')
    if is_vul:
        local_file_name = f'{num}-old{extention}'
    else:
        local_file_name = f'{num}-new{extention}'

    if not cwe_id.lower().find('cwe') > -1:
        cwe_id = 'Other'

    parent_folder = Path(output_folder).absolute()
    logger.debug(f'parent_folder {str(parent_folder)}')
    if not parent_folder.exists():
        parent_folder.mkdir(parents=True)
    cwe_path = Path(parent_folder, cwe_id)
    logger.debug(f'cwe_path {str(cwe_path)}')
    if not cwe_path.exists():
        cwe_path.mkdir(parents=True)

    repo_modified_path_name = '--'.join(Path(repo_name).parts)
    repo_path = Path(cwe_path, repo_modified_path_name)
    logger.debug(f'repo_path {str(repo_path)}')
    if not repo_path.exists():
        repo_path.mkdir(parents=True)

    commit_path = Path(repo_path, commit_sha)
    logger.debug(f'commit_path {str(commit_path)}')
    if not commit_path.exists():
        commit_path.mkdir(parents=True)

    full_file = Path(commit_path, local_file_name)
    logger.debug(f'full_path {str(full_file)}')
    with open(full_file, 'wb') as w:
        w.write(binary_text)
    with open(Path(commit_path, 'meta.json'), 'w') as w:
        json.dump(
            {
                "commit_id": commit_sha,
                "cve_id": cve_id,
                "cwe_id": cwe_id
            }, w)


async def html_str(html_str: str) -> List[str]:
    parser = etree.HTMLParser()
    tree = etree.parse(StringIO(html_str), parser)
    xpath = '//div[@class = "TimelineItem-body"]//ol/li/div/p/a[contains(@class, "Link--primary")]'
    commits_found_by_xpath = tree.xpath(xpath)
    href_list = []
    for found_commit in commits_found_by_xpath:
        attrib = found_commit.attrib.get("href")
        if attrib:
            href = attrib.strip()
            href_list.append(href)
    return href_list


async def get_random_auth_headers(tokens):
    username, token = get_random_token(tokens)
    headers = {'Authorization': "token {}".format(token)}
    return headers


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


async def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to output folder")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        default=Path('vul4j/data/vul4j_dataset.csv'),
        help="Path to input vul4j csv")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    df = df[df['codeLink'].str.startswith('https://github.com/')]
    commits = set()
    #iterrows = list(df.iterrows())[0:20]
    for _, item in tqdm(df.iterrows(), total=df.shape[0]):
        #logger.debug(f"Commit_id {item['commit_id']}; {item['commit_id'].replace('github.com', r'api.github.com/repos')}")
        github_commit = item['codeLink'].replace('github.com', r'api.github.com/repos').replace(
            r'/commit/',
            r'/commits/')
        cwe_id = item['CWE ID']
        cve_id = item.get('CVE ID')
        #logger.debug(f'github_commit {github_commit}')
        commits.add((github_commit, cwe_id, cve_id))

    #all_commits = list(commits)
    #commits = list(commits)[0:51]
    chun = list(chunks(list(commits), 30))
    #logger.debug(chun)
    pbar = tqdm(total=len(chun))
    for commits in chun:
        logger.debug(1)
        session = aiohttp.ClientSession()
        try:
            await asyncio.gather(
                *[process_commit(com, cwe_id, cve_id, session, args) for
                  com, cwe_id, cve_id in
                  commits])
            pbar.update()
            await session.close()
        except:
            pbar.update()
            continue
        finally:
            await session.close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
