"""Mock data for integration tests."""


def get_musique_data():
    """Get mock MuSiQue dataset data for testing."""
    return [{
        "id": "2hop__460946_294723",
        "question": "Who is the spouse of the Green performer?",
        "answer": "Miquette Giraudy",
        "answer_aliases": [],
        "paragraphs": [
            {
                "title": "Steve Hillage",
                "paragraph_text": "Steve Hillage is an English musician, best known as a guitarist. \
He performed the album Green.",
                "is_supporting": True
            },
            {
                "title": "Miquette Giraudy",
                "paragraph_text": "Miquette Giraudy is a French keyboardist and vocalist. \
She is married to Steve Hillage.",
                "is_supporting": True
            }
        ],
        "question_decomposition": []
    }]


def get_musique_corpus():
    """Get mock corpus data for testing."""
    return [
        {
            "title": "Steve Hillage",
            "text": "Steve Hillage is an English musician, best known as a guitarist. He performed the album Green."
        },
        {
            "title": "Miquette Giraudy", 
            "text": "Miquette Giraudy is a French keyboardist and vocalist. She is married to Steve Hillage."
        }
    ]
