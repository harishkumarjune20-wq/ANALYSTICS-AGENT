def generate_conversation_response(question):

    q = (
        question
        or ""
    ).lower().strip()

    # =================================================
    # GREETINGS
    # =================================================

    if q in {
        "hi",
        "hello",
        "hey",
        "hai",
        "hii",
        "hiii"
    }:

        return (
            "Hi! 👋 I'm the AI-Powered Analytics Agent. "
            "How can I help you with company analytics?"
        )

    # =================================================
    # GOODBYE
    # =================================================

    if q in {
        "bye",
        "goodbye",
        "good bye",
        "see you",
        "see you later"
    }:

        return (
            "Goodbye! 👋 "
            "Have a great day."
        )

    # =================================================
    # THANKS
    # =================================================

    if q in {
        "thanks",
        "thank you",
        "thankyou",
        "thx",
        "thank u"
    }:

        return (
            "You're welcome! 😊 "
            "Feel free to ask me about your company analytics."
        )

    # =================================================
    # HOW ARE YOU
    # =================================================

    if q == "how are you":

        return (
            "I'm doing great! 😊 "
            "I'm ready to help with your company analytics."
        )

    # =================================================
    # WHO ARE YOU
    # =================================================

    if q == "who are you":

        return (
            "I'm the AI-Powered Analytics Agent. "
            "I analyze company data and help identify "
            "business performance opportunities."
        )

    # =================================================
    # WHAT CAN YOU DO
    # =================================================

    if q == "what can you do":

        return (
            "I can analyze your data environment company data through "
            "AWS Athena, answer analytics questions, analyze "
            "previous results, and provide business improvement "
            "recommendations."
        )

    # =================================================
    # UNSUPPORTED QUESTION
    # =================================================

    return (
        "I'm the AI-Powered Analytics Agent. "
        "I can help with company analytics, business "
        "performance analysis, and recommendations based "
        "on your analytics results."
    )