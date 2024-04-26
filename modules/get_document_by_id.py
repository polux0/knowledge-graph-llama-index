def get_document_by_id(documents, doc_id):
    # Iterate through the documents and return the one with the matching ID
    for document in documents:
        print("Printing from get_document_by_id loop...")
        print("Document:", )
        if document.id_ == doc_id:
            return document
    return None  # Return None if no document matches the given ID