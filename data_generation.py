import pandas as pd
import numpy as np
import random

def generate_data(num_samples=1000):
    print("Generating synthetic dataset...")
    
    data = []
    
    # ---------------- REAL ACCOUNTS (Label = 0) ----------------
    for i in range(num_samples // 2):
        has_profile_pic = np.random.choice([0, 1], p=[0.05, 0.95])
        num_posts = int(np.random.normal(50, 20))
        num_followers = int(np.random.exponential(500))
        num_following = int(np.random.normal(300, 100))
        bio_len = int(np.random.normal(30, 10))
        
        texts = [
            "Just had a great coffee!", "Loving the weather today.",
            "Working on my new project.", "Can't wait for the weekend!",
            "Had an amazing trip with friends.", "Learning Python is fun!",
            "Check out my latest photo.", "Happy birthday to me!",
            "Life is good.", "Exploring new places."
        ]
        text_content = " ".join(random.choices(texts, k=random.randint(1, 3)))
        
        data.append([has_profile_pic, max(0, num_posts), max(0, num_followers),
                     max(0, num_following), max(0, bio_len), text_content, 0])

    # ---------------- FAKE ACCOUNTS (Label = 1) ----------------
    for i in range(num_samples // 2):
        has_profile_pic = np.random.choice([0, 1], p=[0.8, 0.2])
        num_posts = int(np.random.exponential(5))
        num_followers = int(np.random.exponential(20))
        num_following = int(np.random.normal(1000, 200))
        bio_len = int(np.random.exponential(5))
        
        texts = [
            "Click here for free money!", "Follow for follow back.",
            "Best deals on crypto.", "Earn $500 daily easily!",
            "Win a free iPhone now.", "Cheap luxury bags.",
            "Visit my profile for more.", "Make money online fast.",
            "Discount codes available.", "DM for collaboration."
        ]
        text_content = " ".join(random.choices(texts, k=random.randint(1, 3)))
        
        data.append([has_profile_pic, max(0, num_posts), max(0, num_followers),
                     max(0, num_following), max(0, bio_len), text_content, 1])

    columns = [
        'has_profile_pic', 'num_posts', 'num_followers',
        'num_following', 'bio_len', 'text_content', 'is_fake'
    ]

    # Create DataFrame & shuffle
    df = pd.DataFrame(data, columns=columns)
    df = df.sample(frac=1).reset_index(drop=True)

    # ---------------- MANUAL FAKE ACCOUNT (ALWAYS LAST) ----------------
    manual_fake_account = pd.DataFrame([[
        0, 1, 5, 3000, 2, "Earn money fast click here", 1
    ]], columns=columns)

    df = pd.concat([df, manual_fake_account], ignore_index=True)

    output_file = 'fake_accounts_dataset.csv'
    df.to_csv(output_file, index=False)

    print("Manual fake account added as LAST ROW.")
    print(f"Dataset generated and saved to {output_file}")

if __name__ == "__main__":
    generate_data()
