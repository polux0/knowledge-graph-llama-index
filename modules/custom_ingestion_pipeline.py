from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import (
    BaseComponent,
    BaseNode,
    Document,
    MetadataMode,
    TransformComponent,
)
from typing import Any, List, Optional, Sequence
from enum import Enum
from llama_index.core.ingestion.cache import DEFAULT_CACHE_NAME, IngestionCache
from hashlib import sha256
import re
from ratelimit import limits, sleep_and_retry
from functools import wraps
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class DocstoreStrategy(str, Enum):
    """Document de-duplication strategy."""

    UPSERTS = "upserts"
    DUPLICATES_ONLY = "duplicates_only"
    UPSERTS_AND_DELETE = "upserts_and_delete"


class CustomIngestionPipeline(IngestionPipeline):
    def rate_limiter(self, calls: int, period: int):
        def decorator(func):
            @wraps(func)
            @sleep_and_retry
            @limits(calls=calls, period=period)
            def wrapper(*args, **kwargs):
                logging.debug(f"Rate limited function {func.__name__} called with {calls} calls per {period} seconds")
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def remove_unstable_values(self, s: str) -> str:
        """Remove unstable key/value pairs.

        Examples include:
        - <__main__.Test object at 0x7fb9f3793f50>
        - <function test_fn at 0x7fb9f37a8900>
        """
        pattern = r"<[\w\s_\. ]+ at 0x[a-z0-9]+>"
        return re.sub(pattern, "", s)
    def _prepare_inputs(
        self, documents: Optional[List[Document]], nodes: Optional[List[BaseNode]]
    ) -> List[Document]:
        input_nodes: List[BaseNode] = []
        if documents is not None:
            input_nodes += documents

        if nodes is not None:
            input_nodes += nodes

        if self.documents is not None:
            input_nodes += self.documents

        if self.readers is not None:
            for reader in self.readers:
                input_nodes += reader.read()

        return input_nodes
    def get_transformation_hash(
        self,
        nodes: List[BaseNode], transformation: TransformComponent
    ) -> str:
        """Get the hash of a transformation."""
        nodes_str = "".join(
            [str(node.get_content(metadata_mode=MetadataMode.ALL)) for node in nodes]
        )

        transformation_dict = transformation.to_dict()
        transform_string = self.remove_unstable_values(str(transformation_dict))

        return sha256((nodes_str + transform_string).encode("utf-8")).hexdigest()

    def run(
        self,
        show_progress: bool = False,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[BaseNode]] = None,
        cache_collection: Optional[str] = None,
        in_place: bool = True,
        store_doc_text: bool = True,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        input_nodes = self._prepare_inputs(documents, nodes)

        if self.docstore is not None:
            if self.docstore_strategy == DocstoreStrategy.UPSERTS_AND_DELETE or \
               self.docstore_strategy == DocstoreStrategy.UPSERTS:
                nodes_to_run = self._handle_upserts(input_nodes, store_doc_text)
            elif self.docstore_strategy == DocstoreStrategy.DUPLICATES_ONLY:
                nodes_to_run = self._handle_duplicates(input_nodes, store_doc_text)
            else:
                raise ValueError(f"Invalid docstore strategy: {self.docstore_strategy}")
        else:
            nodes_to_run = input_nodes

        if num_workers and num_workers > 1:
            nodes = self._process_in_parallel(nodes_to_run, num_workers, cache_collection, in_place, **kwargs)
        else:
            nodes = self.run_transformations_with_rate_limit(
                nodes_to_run,
                transformations=self.transformations,
                show_progress=show_progress,
                cache=self.cache if not self.disable_cache else None,
                cache_collection=cache_collection,
                in_place=in_place,
                **kwargs,
            )

        if self.vector_store is not None:
            self.vector_store.add([n for n in nodes if n.embedding is not None])

        return nodes

    def run_transformations(
        self,
        nodes: List[BaseNode],
        transformations: Sequence[TransformComponent],
        in_place: bool = True,
        cache: Optional[IngestionCache] = None,
        cache_collection: Optional[str] = None,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Run a series of transformations on a set of nodes.

        Args:
            nodes: The nodes to transform.
            transformations: The transformations to apply to the nodes.

        Returns:
            The transformed nodes.
        """
        if not in_place:
            nodes = list(nodes)

        for transform in transformations:
            if cache is not None:
                hash = self.get_transformation_hash(nodes, transform)
                cached_nodes = cache.get(hash, collection=cache_collection)
                if cached_nodes is not None:
                    nodes = cached_nodes
                else:
                    nodes = transform(nodes, **kwargs)
                    cache.put(hash, nodes, collection=cache_collection)
            else:
                nodes = transform(nodes, **kwargs)

        return nodes

    # def _modify_transformations(self, transformations):
    #     # Potentially modify or filter the transformations
    #     return transformations

    def run_transformations_with_rate_limit(
            self,
            nodes,
            transformations,
            show_progress,
            cache,
            cache_collection,
            in_place,
            rate_calls=5,
            rate_period=60,
            **kwargs):
        decorated_run_transformations = self.rate_limiter(rate_calls,rate_period)(self.run_transformations)
        logging.info("Applying custom transformations with rate limiting...")
        return decorated_run_transformations(
            nodes,
            transformations,
            show_progress=show_progress,
            cache=cache,
            cache_collection=cache_collection,
            in_place=in_place,
            **kwargs
        )