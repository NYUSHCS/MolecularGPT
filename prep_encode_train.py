"""Fingerprints the molecules, encoding them with 4 techniques, producing 28 Billions fingerprints for 7 Billion molecules."""

import os
import logging
from typing import List, Callable
from multiprocessing import Process, cpu_count
from rdkit.Chem import AllChem, MACCSkeys


import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from usearch.index import Index, CompiledMetric, MetricKind, MetricSignature, ScalarKind
from usearch.eval import self_recall, SearchStats

from metrics_numba import (
    tanimoto_conditional,
    tanimoto_maccs,
)
from to_fingerprint import (
    # smiles_to_maccs_ecfp4_fcfp4,
    # smiles_to_pubchem,
    shape_mixed,
    shape_maccs,
)

from dataset import (
    write_table,
    FingerprintedDataset,
    FingerprintedEntry,
)

logger = logging.getLogger(__name__)

from rdkit import Chem
def molecule_to_maccs(
    smiles: str):

    molecule = Chem.MolFromSmiles(smiles)

    return np.packbits(MACCSkeys.GenMACCSKeys(molecule))


def augment_with_rdkit(parquet_path: os.PathLike):
    
    meta = pq.read_metadata(parquet_path)
    column_names: List[str] = meta.schema.names
    if "maccs" in column_names:
        return

    logger.info(f"Starting file {parquet_path}")
    table: pa.Table = pq.read_table(parquet_path)
    maccs_list = []

    for smiles in table["smiles"]:
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
    print(parquet_dir, augmentation, shard_index, shards_count)
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
            print(parquet_dir, augmentation, i, processes)
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


def shards_index(dataset: FingerprintedDataset):
    os.makedirs(os.path.join(dataset.dir, "usearch-maccs"), exist_ok=True)

    for shard_idx, shard in enumerate(dataset.shards):
        index_path_maccs = os.path.join(
            dataset.dir, "usearch-maccs", shard.name + ".usearch"
        )


        if (
            Index.metadata(index_path_maccs) is not None
        ):
            continue
        logger.info(f"Starting {shard_idx + 1} / {len(dataset.shards)}")
        table = shard.load_table()
        n = len(table)

        # No need to shuffle the entries as they already are:

        keys = np.arange(shard.first_key, shard.first_key + n)
        maccs_fingerprints = [table["maccs"][i].as_buffer() for i in range(n)]


        # First construct the index just for MACCS representations
        vectors = np.vstack(
            [
                FingerprintedEntry.from_parts(
                    None,
                    maccs_fingerprints[i],
                    None,
                    None,
                    shape_maccs,
                ).fingerprint
                for i in range(n)
            ]
        )

        index_maccs = Index(
            ndim=shape_maccs.nbits,
            dtype=ScalarKind.B1,
            metric=CompiledMetric(
                pointer=tanimoto_maccs.address,
                kind=MetricKind.Tanimoto,
                signature=MetricSignature.ArrayArray,
            ),
        )
        index_maccs.add(
            keys,
            vectors,
            log=f"Building {index_path_maccs}",
            batch_size=100_000,
        )

        # Optional self-recall evaluation:
        stats: SearchStats = self_recall(index_maccs, sample=0.01)
        logger.info(f"Self-recall: {100*stats.mean_recall:.2f} %")
        logger.info(f"Efficiency: {100*stats.mean_efficiency:.2f} %")
        index_maccs.save(index_path_maccs)


        # Discard the objects to save some memory
        dataset.shards[shard_idx].table_cached = None
        dataset.shards[shard_idx].index_cached = None


def mono_index_maccs(dataset: FingerprintedDataset):
    index_path_maccs = os.path.join("indexes", dataset.dir, "usearch-maccs.usearch")
    print(index_path_maccs)
    os.makedirs(os.path.join("indexes", dataset.dir), exist_ok=True)

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
                        None,
                        maccs_fingerprints[i],
                        None,
                        None,
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
    logger.info("Time to encode some molecules!")

    processes = max(cpu_count() - 4, 1)

    datasets = ['assay','property','qm9']
    
    for dataset in datasets:
        augment_parquet_shards(os.path.join("./train_process",dataset), augment_with_rdkit, processes)

