"""Explicit supplemental TRAIN examples for the missing unsubscribe operation.

Category NEWSLETTER for unsubscribe is already expected in the original 15-item
demo. These are handwritten training additions, not real customers or tests.
They are not used to claim performance on unseen intents in LOIO.
"""
EXAMPLES = [
    {'text':text,'category':'NEWSLETTER','intent':'extension_newsletter_unsubscribe',
     'source':'handwritten_training_extension','sample_weight':3.0}
    for text in [
        'Unsubscribe my email from the newsletter subscription.',
        'I would prefer not to receive advertisements in my inbox.',
        'Cancel the subscription to your email bulletin.',
        'Opt me out of advertising emails from the shop.',
        'I no longer wish to receive weekly offers by email.',
        'Turn off commercial email notifications, but keep my customer account.',
        'The advertising mailing is unwanted; please unsubscribe this subscriber.',
        'Disable the email subscription for promotions and special offers.',
    ]
]

def eligible_examples(frame, leave_intent_out=False):
    # Every LOIO run excludes the extension, preserving the original definition
    # of intent holdout and avoiding teaching a withheld task through new data.
    if leave_intent_out:
        return []
    categories = set(frame.category)
    return [x for x in EXAMPLES if x['category'] in categories]
