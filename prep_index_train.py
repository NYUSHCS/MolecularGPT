"""Indexes fingerprints using USearch."""

import os
import logging
from typing import List, Callable
from multiprocessing import Process, cpu_count
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from usearch.index import Index, CompiledMetric, MetricKind, MetricSignature, ScalarKind
from usearch.eval import self_recall, SearchStats

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

from metrics_numba import (
    tanimoto_conditional,
    tanimoto_mixed,
    tanimoto_maccs,
)
from dataset import (
    write_table,
    FingerprintedDataset,
    FingerprintedEntry,
)
from to_fingerprint import (
    smiles_to_maccs_ecfp4_fcfp4,
    smiles_to_pubchem,
    shape_mixed,
    shape_maccs,
)


def molecule_to_maccs(
    smiles: str):
    # print(smiles)
    molecule = Chem.MolFromSmiles(smiles)
    # print(molecule)
    # print(MACCSkeys.GenMACCSKeys(molecule))
    return np.packbits(MACCSkeys.GenMACCSKeys(molecule))



logger = logging.getLogger(__name__)


def augment_with_rdkit(parquet_path: os.PathLike):
    meta = pq.read_metadata(parquet_path)
    column_names: List[str] = meta.schema.names
    if "maccs" in column_names:
        return

    logger.info(f"Starting file {parquet_path}")
    table: pa.Table = pq.read_table(parquet_path)

    maccs_list = []

    for smiles in table["graph"]:
        try:
            fingers = molecule_to_maccs(str(smiles))
            maccs_list.append(fingers.tobytes())
        except Exception:
            maccs_list.append(bytes(bytearray(21)))

    maccs_list = pa.array(maccs_list, pa.binary(21))

    maccs_field = pa.field("maccs", pa.binary(21), nullable=False)

    table = table.append_column(maccs_field, maccs_list)
    write_table(table, parquet_path)


def augment_parquets_shard(
    parquet_dir: os.PathLike,
    augmentation: Callable,
    shard_index: int,
    shards_count: int,
):
    filenames: List[str] = sorted(os.listdir(parquet_dir))
    files_count = len(filenames)
    try:
        for file_idx in range(shard_index, files_count, shards_count):
            try:
                filename = filenames[file_idx]
                augmentation(os.path.join(parquet_dir, filename))
                logger.info(
                    "Augmented shard {}. Process # {} / {}".format(
                        filename, shard_index, shards_count
                    )
                )
            except KeyboardInterrupt as e:
                raise e

    except KeyboardInterrupt as e:
        logger.info(f"Stopping shard {shard_index} / {shards_count}")
        raise e


def augment_parquet_shards(
    parquet_dir: os.PathLike,
    augmentation: Callable,
    processes: int = 1,
):
    if processes > 1:
        process_pool = []
        for i in range(processes):
            p = Process(
                target=augment_parquets_shard,
                args=(parquet_dir, augmentation, i, processes),
            )
            p.start()
            process_pool.append(p)

        for p in process_pool:
            p.join()
    else:
        augment_parquets_shard(parquet_dir, augmentation, 0, 1)


def mono_index_maccs(dataset: FingerprintedDataset):

    os.makedirs(os.path.join(dataset.dir), exist_ok=True)

    index_maccs = Index(
        ndim=shape_maccs.nbits,
        dtype=ScalarKind.B1,
        metric=CompiledMetric(
            pointer=tanimoto_maccs.address,
            kind=MetricKind.Tanimoto,
            signature=MetricSignature.ArrayArray,
        ),
        # path=index_path_maccs,
    )

    try:
        for shard_idx, shard in enumerate(dataset.shards):
            if shard.name.split('_')[0] == 'test':
                continue
            index_path_maccs = os.path.join(dataset.dir, "index-maccs.usearch")

            if shard.first_key in index_maccs:
                logger.info(f"Skipping {shard_idx + 1} / {len(dataset.shards)}")
                continue

            logger.info(f"Starting {shard_idx + 1} / {len(dataset.shards)}")
            table = shard.load_table(["maccs"])
            n = len(table)

            # No need to shuffle the entries as they already are:
            keys = np.arange(shard.first_key, shard.first_key + n)
            maccs_fingerprints = [table["maccs"][i].as_buffer() for i in range(n)]

            # First construct the index just for MACCS representations
            vectors = np.vstack(
                [
                    FingerprintedEntry.from_parts(
                        # None,
                        maccs_fingerprints[i],
                        # None,
                        # None,
                        shape_maccs,
                    ).fingerprint
                    for i in range(n)
                ]
            )

            index_maccs.add(keys, vectors, log=f"Building {index_path_maccs}")

            # Optional self-recall evaluation:
            # stats: SearchStats = self_recall(index_maccs, sample=1000)
            # logger.info(f"Self-recall: {100*stats.mean_recall:.2f} %")
            # logger.info(f"Efficiency: {100*stats.mean_efficiency:.2f} %")
            if shard_idx % 100 == 0:
                index_maccs.save(index_path_maccs)

            # Discard the objects to save some memory
            dataset.shards[shard_idx].table_cached = None
            dataset.shards[shard_idx].index_cached = None

        index_maccs.save(index_path_maccs)
        index_maccs.reset()
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
    logger.info("Time to index some molecules!")

    processes = max(cpu_count() - 4, 1)
    
    # train datasets
    path ='./train_process'
    datasets = ['assay','property','qm9']

    
    for dataset in datasets:
        path_1 = os.path.join(path, dataset)

        file_names = [f.name for f in Path(path_1).iterdir() if f.is_dir()]
        for f in file_names:
            mono_index_maccs(FingerprintedDataset.open((os.path.join(path_1, f))))
            
            
    