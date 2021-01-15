from indexing import index

# Indexation de toute les bases
from search import search

index("Copydays")
search("Copydays", "Copydays_original_200000_512", "KDTREE", 7, 10)