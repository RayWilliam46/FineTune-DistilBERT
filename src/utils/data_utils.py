def combine_toxic_classes(df):
    """""""""
    Reconfigures the Jigsaw Toxic Comment dataset from a 
    multi-label classification problem to a
    binary classification problem predicting if a text is
    toxic (class=1) or non-toxic (class=0).

    Input:
        - df:  A pandas DataFrame with columns:
               - 'id'
               - 'comment_text'
               - 'toxic'
               - 'severe_toxic'
               - 'obscene'
               - 'threat'
               - 'insult'
               - 'identity_hate'
    Output:
        - df:  A modified pandas DataFrame with columns:
               - 'comment_text' containing strings of text.
               - 'isToxic' binary target variable containing 0's and 1's.
    """""""""
    
    # Create a binary classification label for 'isToxic'
    # and drop miscellaneous labels.
    df['isToxic'] = (df['toxic'] == 1)
    drop_cols = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df.drop(columns=drop_cols, inplace=True)
    df.replace(to_replace={'isToxic': {True: 1, False: 0}}, inplace=True)

    # Cast column values to save memory
    df['isToxic'] = df['isToxic'].astype('int8')

    return df



def undersample_majority(df, percent_conserve):
    """""""""
    Undersamples the majority class of the Jigsaw Toxic Comment dataset
    ('isToxic'==0) by conserving a given percent of the majority class.

    Inputs:
        - df:  A pandas DataFrame with columns:
               - 'comment_text' containing strings of text.
               - 'isToxic' binary target variable containing 0's and 1's. 
        - percent_conserve:  Float representing fraction of 
                             majority class (clean_texts) to conserve
    Outputs:
        - downsampled_df:    A new pandas DataFrame that has been shuffled
                             and has had its majority class downsampled.
    """""""""
    
    # Get rows of clean and toxic texts
    clean_texts = df[df['isToxic'] == 0]
    toxic_texts = df[df['isToxic'] == 1]

    # Randomly sample from the majority class and construct a new DataFrame
    # consisting of the majority class (clean_texts) + the minority classes (toxic_texts)
    to_conserve = clean_texts.sample(frac=percent_conserve, random_state=42)
    downsampled_df = to_conserve.append(toxic_texts, ignore_index=True)

    return downsampled_df.sample(frac=1, random_state=42).reset_index(drop=True)



def analyze_dist(df):
    """""""""
    Analyzes the class distribution of a pandas DataFrame.

    Input:
        - df:  a pandas DataFrame containing text whose toxicity is denoted
               by the 'isToxic' binary indicator column.
    Output:
        - Prints class distribution (toxic or non-toxic) statistics of df.
    """""""""
    
    print('Total rows:           ', df.shape[0])
    print('Clean texts:          ', df.shape[0] - df['isToxic'].sum())
    print('Toxic texts:          ', df['isToxic'].sum())
    print('Toxic texts make up   ', ((df['isToxic'].sum() / df.shape[0]) * 100).round(2), 'percent of our total data')
    return



def get_relevant_words(text, to_conserve):
    """""""""
    Takes a string of text and returns the first N words in that text.
    
    Input:
        - text:         String of text
        - to_conserve:  Integer representing number of text's words to conserve
    Output:
        - String containing first (to_conserve) words of text. 
    """""""""
    
    # Select the first N words in the text
    word_list = text.split()[:to_conserve]
    
    # Build up a string containing words in word_list
    new_string = ' '.join(word for word in word_list)
    
    return new_string



def augment_sentence(sentence, aug, num_threads):
    """""""""
    Constructs a new sentence via text augmentation.

    Input:
        - sentence:     A string of text
        - aug:          An augmentation object defined by the nlpaug library
        - num_threads:  Integer controlling the number of threads to use if
                        augmenting text via CPU
    Output:
        - A string of text that been augmented
    """""""""
    return aug.augment(sentence, num_thread=num_threads)



def augment_text(df, aug, num_threads, num_times):
    """""""""
    Takes a pandas DataFrame and augments its text data.

    Input:
        - df:            A pandas DataFrame containing the columns:
                                - 'comment_text' containing strings of text to augment.
                                - 'isToxic' binary target variable containing 0's and 1's.
        - aug:           Augmentation object defined by the nlpaug library.
        - num_threads:   Integer controlling number of threads to use if augmenting
                         text via CPU
        - num_times:     Integer representing the number of times to augment text.
    Output:
        - df:            Copy of the same pandas DataFrame with augmented data 
                         appended to it and with rows randomly shuffled.
    """""""""
    
    # Get rows of data to augment
    to_augment = df[df['isToxic'] == 1]
    to_augmentX = to_augment['comment_text']
    to_augmentY = np.ones(len(to_augmentX.index) * num_times, dtype=np.int8)

    # Build up dictionary containing augmented data
    aug_dict = {'comment_text': [], 'isToxic': to_augmentY}
    for i in tqdm(range(num_times)):
        augX = [augment_sentence(x, aug, num_threads) for x in to_augmentX]
        aug_dict['comment_text'].extend(augX)

    # Build DataFrame containing augmented data
    aug_df = pd.DataFrame.from_dict(aug_dict)

    return df.append(aug_df, ignore_index=True).sample(frac=1, random_state=42)
