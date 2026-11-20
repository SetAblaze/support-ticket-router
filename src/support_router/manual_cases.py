"""Author-written category diagnostics, separate from the supplied Bitext CSV.

DEV is a development set used when choosing a model; HOLDOUT was written before
the extension was tested and is used only after fixing the model and policy.
Neither set is a customer sample. No real-world accuracy claim follows.
"""
import pandas as pd

DEV = {
    'ACCOUNT': ['I forgot my password and cannot sign in to my profile',
                'Please close my customer account permanently','How do I register as a new customer?'],
    'CANCELLATION_FEE': ['Will I be charged a penalty if I cancel my purchase?',
                'Is there a fee for cancelling a booking?','How much does cancellation cost?'],
    'CONTACT': ['Please connect me to a customer service representative',
                'How can I speak with someone who works in support?',
                'What telephone number should I use to reach your help desk?'],
    'DELIVERY': ['Do you offer express delivery?','How many days does standard delivery usually take?',
                'Which courier delivery options are available before I order?'],
    'FEEDBACK': ['I would like to leave a positive review of your service',
                'Your staff were rude and I want to file a complaint',
                'I am really disappointed with the quality of your service'],
    'INVOICES': ['Please email me an invoice for my purchase','I need a copy of the bill for my expense report',
                'Where can I download my past invoices?'],
    'NEWSLETTER': ['Stop sending me marketing emails','Please remove me from your promotional mailing list',
                'I would like to subscribe to your newsletter'],
    'ORDER': ['My parcel has not arrived; where is my order now?','Please cancel the order I placed this morning',
                'I want to add another item to my existing order'],
    'PAYMENT': ['My card keeps getting declined when I try to pay','Can I use PayPal to pay for my purchase?',
                'Which payment methods do you accept?'],
    'REFUNDS': ['I sent the item back but have not received my refund','Am I eligible to get my money back?',
                'Please tell me the status of my reimbursement'],
    'SHIPPING': ['Please send future packages to my new address',
                'I entered the wrong shipping address and need to correct it',
                'Where do I save a delivery address in my profile?'],
}

HOLDOUT = {
    'ACCOUNT': ['My login credentials no longer work and I need to reset the password.',
                'I want to replace the email associated with my customer profile.',
                'Please erase my registered user account.'],
    'CANCELLATION_FEE': ['Before I cancel, tell me the amount of the cancellation penalty.',
                'Do you deduct a cancellation charge when a booking is called off?',
                'What is the fee for cancelling my reservation?'],
    'CONTACT': ['I need a phone number for the customer care team.',
                'Could you transfer this chat to a real person from support?',
                'How do I reach a human advisor instead of the bot?'],
    'DELIVERY': ['Before placing a purchase, I want to know the usual delivery timeframe.',
                'Can customers choose next-day delivery instead of the standard service?',
                'Which delivery services can I choose at checkout?'],
    'FEEDBACK': ['The assistant was extremely helpful and I would like to post a review.',
                'I am writing to complain about the treatment I received from your staff.',
                'The service was disappointing; where can I submit my feedback?'],
    'INVOICES': ['Could you supply the billing document for my purchase for accounting purposes?',
                'I need a duplicate of the invoice that was issued last month.',
                'How can I view the invoices attached to previous transactions?'],
    'NEWSLETTER': ['Please take my email off the list used for advertising campaigns.',
                'Your sale announcements are filling my inbox and I want them to end.',
                'Sign me up to receive the store newsletter.'],
    'ORDER': ['I placed a purchase last week; can you locate the parcel for me?',
                'I ordered the wrong quantity and want to amend the items in my purchase.',
                'Please call off the purchase I submitted a few minutes ago.'],
    'PAYMENT': ['The checkout rejects my debit card whenever I attempt payment.',
                'Is paying by bank transfer supported by your store?',
                'The payment failed, although my card has sufficient funds.'],
    'REFUNDS': ['The reimbursement was approved but the money has not reached my bank yet.',
                'Please explain the conditions for receiving my money back.',
                'I would like to request reimbursement for the returned purchase.'],
    'SHIPPING': ['The destination address saved for deliveries is outdated; how can I replace it?',
                'I need to enter my home address as a new shipping destination.',
                'Please correct the house number in the delivery address on my purchase.'],
}

def frames():
    def convert(cases,role):
        return pd.DataFrame([{'text':text,'category':category,'role':role,
                              'source':'author_written','used_for_training':False}
                             for category,texts in cases.items() for text in texts])
    return convert(DEV,'model_selection'),convert(HOLDOUT,'post_selection_diagnostic')
