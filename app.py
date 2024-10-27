from flask import Flask, render_template, request, redirect, url_for
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from datetime import datetime
import re
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'

class EmotionalAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.history = []

        
        self.emotion_keywords = {
            # Positive emotions 
            'Happiness': {
                'polarity': 1,
                'keywords': ['good', 'joy', 'delight', 'pleasure', 'cheerful', 'bliss', 'elated', 'radiant', 'smiling', 
                'grinning', 'content', 'satisfied', 'overjoyed', 'happy', 'upbeat', 'gleeful', 'good mood']
            },
            'Gratitude': {
                'polarity': 1,
                'keywords': ['thankful', 'grateful', 'appreciative', 'blessed', 'recognizing blessings', 'honored', 
                'indebted', 'gratitude', 'showing thanks', 'giving thanks', 'feeling fortunate', 'appreciating']
            },
            'Love': {
                'polarity': 1,
                'keywords': ['affection', 'adoration', 'caring', 'fondness', 'intimacy', 'cherishing', 'devotion', 
                'warmth', 'infatuated', 'attachment', 'compassionate', 'connected', 'appreciation', 'adoring']
            },
            'Excitement': {
                'polarity': 1,
                'keywords': ['fantastic', 'excited', 'thrilled', 'eager', 'anticipation', 'enthusiastic', 'energized', 'animated', 'pumped', 
                'buzzing', 'fired up', 'elated', 'looking forward', 'enthusiastic', "can't wait", 'adrenaline']
            },
            'Contentment': {
                'polarity': 1,
                'keywords': ['satisfied', 'at ease', 'peaceful', 'serene', 'tranquil', 'feeling good', 'comfortable', 
                'fulfilled', 'relaxed', 'in harmony', 'balanced', 'untroubled', 'at peace', 'blissful']
            },
            'Hope': {
                'polarity': 1,
                'keywords': ['optimistic', 'hopeful', 'looking forward', 'positive outlook', 'expecting the best', 
                'encouraged', 'bright future', 'believing', 'trusting', 'faithful', 'confident']
            },
            'Pride': {
                'polarity': 1,
                'keywords': ['accomplished', 'proud', 'successful', 'fulfilled', 'self-worth', 'confident', 
                'dignified', 'achieved', 'boosted', 'self-assured', 'satisfied with self', 'valued']
            },
            'Compassion': {
                'polarity': 1,
                'keywords': ['caring', 'empathetic', 'sympathetic', 'understanding', 'supportive', 'kind', 
                'helpful', 'nurturing', 'consoling', 'altruistic', 'selfless', 'generous', 'forgiving']
            },
            'Confidence': {
                'polarity': 1,
                'keywords': ['self-assured', 'bold', 'fearless', 'empowered', 'assertive', 'secure', 'sure of oneself', 
                'competent', 'trusting oneself', 'capable', 'determined', 'decisive', 'clear-minded']
            },
            'Interest': {
                'polarity': 1,
                'keywords': ['curious', 'intrigued', 'engaged', 'absorbed', 'fascinated', 'attentive', 'inspired', 
                'invested', 'motivated', 'keen', 'open-minded', 'enthused', 'inquiring']
            },
            'Relief': {
                'polarity': 1,
                'keywords': ['calm', 'reassured', 'comforted', 'at ease', 'soothed', 'unburdened', 'lightened', 
                'relaxed', 'free from worry', 'peaceful', 'let go of stress', 'untroubled']
            },
            'Inspiration': {
                'polarity': 1,
                'keywords': ['motivated', 'encouraged', 'uplifted', 'driven', 'moved', 'aspirational', 'influenced', 
                'passionate', 'awakened', 'stimulated', 'sparked', 'creative energy', 'rejuvenated']
            },

            # Neutral emotions 
            'Calm': {
                'polarity': 0,
                'keywords': ['at peace', 'tranquil', 'relaxed', 'unruffled', 'quiet', 'still', 'collected', 
                'composed', 'unbothered', 'serene', 'untroubled', 'even-tempered']
            },
            'Content': {
                'polarity': 0,
                'keywords': ['satisfied', 'fine', 'okay', 'moderate', 'comfortable', 'balanced', 'unperturbed', 
                'at ease', 'mellow', 'settled', 'coping', 'decent']
            },
            'Indifferent': {
                'polarity': 0,
                'keywords': ['meh', 'eh', 'unconcerned', 'detached', 'neutral', 'disinterested', 'unmoved', 'nonchalant', 
                'apathetic', 'unaffected', 'impartial', 'dispassionate']
            },
            'Thoughtful': {
                'polarity': 0,
                'keywords': ['pensive', 'reflective', 'contemplative', 'in thought', 'introspective', 'meditative', 
                'considerate', 'musing', 'ruminative', 'mindful']
            },
            'Uncertain': {
                'polarity': 0,
                'keywords': ['undecided', 'unsure', 'ambivalent', 'doubtful', 'hesitant', 'wavering', 'tentative', 
                'in-between', 'conflicted', 'in limbo']
            },
            'Reserved': {
                'polarity': 0,
                'keywords': ['restrained', 'guarded', 'cautious', 'conservative', 'mild-mannered', 'unassuming', 
                'introverted', 'modest', 'aloof', 'subdued']
            },
            'Bored': {
                'polarity': 0,
                'keywords': ['bored', 'uninterested', 'disengaged', 'apathetic', 'uninspired', 'listless', 'indifferent', 
                'unmotivated', 'lacking excitement', 'bland', 'routine']
            },
            'Acceptance': {
                'polarity': 0,
                'keywords': ['agree', 'accepting', 'allowing', 'yielding', 'compliant', 'receptive', 'acknowledging', 
                'resigned', 'agreeable', 'accommodating', 'tolerant']
            },

            # Negative emotions 
            
            'Insomnia': {
                'polarity': -1,
                'keywords': ["can't sleep", 'sleepless', 'awake', 'insomnia', 'tired', 'fatigue', 'restless', 
                'alert', 'unable to sleep', 'night owl', 'exhausted']
            },
            'Fear': {
                'polarity': -1,
                'keywords': ['scared', 'afraid', 'terrified', 'fearful', 'panic', 'dread', 'horrified', 
                'worried', 'uneasy', 'shaken', 'startled']
            },
            'Anger': {
                'polarity': -1,
                'keywords': ['angry', 'furious', 'mad', 'rage', 'irritated', 'frustrated', 'annoyed', 
                'resentful', 'offended', 'infuriated', 'aggravated']
            },
            'Anxiety Disorder': {
                'polarity': -1,
                'keywords': ['anxious', 'nervous', 'panic', 'constant worry', 'uneasy', 'fearful', 'restless', 
                'jittery', 'tense', 'racing thoughts', 'overthinking']
            },
            'Depression': {
                'polarity': -1,
                'keywords': ['sad', 'hopeless', 'depressed', 'down', 'empty', 'worthless', 'no motivation', 
                'fatigue', 'loss of interest', 'guilt', 'difficulty concentrating']
            },

            'Bipolar Disorder': {
                'polarity': -1,
                'keywords': ['manic', 'high energy', 'euphoric', 'extremely happy', 'depressive episode', 'mood swings', 
                'impulsive', 'hyperactive', 'low energy', 'reckless behavior', 'irritable']
            },

            'PTSD': {
                'polarity': -1,
                'keywords': ['trauma', 'flashbacks', 'nightmares', 'hypervigilant', 'distressing memory', 'startle', 
                'panic attack', 'emotional numbness', 'detached', 'intrusive thoughts']
            },

            'ADHD': {
                'polarity': -1,
                'keywords': ['easily distracted', 'hyperactive', 'restless', 'impulsive', 'difficulty focusing', 
                'inattentive', 'short attention span', 'disorganized', 'forgetful', 'fidgeting']
            },

            'Schizophrenia': {
                'polarity': -1,
                'keywords': ['hallucinations', 'delusions', 'paranoia', 'disorganized thoughts', 'hearing voices', 
                'psychosis', 'detachment from reality', 'social withdrawal', 'agitation', 'irrational beliefs']
            },

            'OCD': {
                'polarity': -1,
                'keywords': ['obsessive thoughts', 'compulsions', 'rituals', 'checking behavior', 'repeated actions', 
                'fear of contamination', 'excessive cleaning', 'repetitive thoughts', 'intrusive thoughts', 'perfectionism']
            },

            'Eating Disorder': {
                'polarity': -1,
                'keywords': ['obsese', 'body image', 'weight loss', 'restricting food', 'binge eating', 'purging', 'anorexia', 
                'bulimia', 'self-starvation', 'over-exercising', 'preoccupation with weight']
            },

            'Substance Abuse': {
                'polarity': -1,
                'keywords': ['addiction', 'substance use', 'alcohol abuse', 'drug dependence', 'cravings', 'withdrawal', 
                'overdose', 'self-medicate', 'rehab', 'relapse', 'intoxicated']
            },

            'Sleep Disorder': {
                'polarity': -1,
                'keywords': ['insomnia', 'night terrors', 'sleepwalking', 'sleep apnea', 'difficulty sleeping', 'fatigue', 
                'daytime sleepiness', 'oversleeping', 'restless sleep', 'hypersomnia']
            },

            'Stress': {
                'polarity': -1,
                'keywords': ['stress', 'overwhelmed', 'tense', 'pressure', 'strained', 'burned out', 'anxious', 
                'stressed out', 'under pressure', 'nervous tension']
            },

            'Phobia': {
                'polarity': -1,
                'keywords': ['irrational fear', 'extreme fear', 'panic', 'phobia', 'avoidance', 'specific fear', 
                'afraid of heights', 'claustrophobia', 'social anxiety', 'fearful of situations']
            },

            'Anger Issue': {
                'polarity': -1,
                'keywords': ['anger issue', 'rage', 'outburst', 'easily angered', 'irritability', 'temper', 
                'explosive anger', 'hostility', 'short-tempered', 'agitated']
            },

            'Loneliness': {
                'polarity': -1,
                'keywords': ['lonely', 'isolated', 'alone', 'socially withdrawn', 'unwanted solitude', 'alienated', 
                'feeling left out', 'disconnected', 'lack of companionship', 'social isolation']
            },

            'Suicidal Ideation': {
                'polarity': -1,
                'keywords': ['suicidal thoughts', 'want to end life', 'thinking about death', 'hopelessness', 
                'no reason to live', 'suicidal plans', 'self-harm thoughts', 'life is pointless', 'depressed']
            },

            'Self Harm': {
                'polarity': -1,
                'keywords': ['self-harm', 'cutting', 'burning', 'harming oneself', 'scratching', 'self-injury', 
                'self-destruction', 'physically hurting oneself', 'self-inflicted pain']
            },

            'Panic Attack': {
                'polarity': -1,
                'keywords': ['panic attack', 'intense fear', 'heart racing', 'chest pain', 'shortness of breath', 
                'dizzy', 'shaking', 'sweating', 'sudden terror', 'fear of losing control']
            },

            'Sadness': {
                'polarity': -1,
                'keywords': ['sad', 'crying', 'tears', 'depressed', 'unhappy', 'miserable', 'lonely',
                           'heartbroken', 'down', 'blue'],
            }

        }

    def analyze_text(self, text, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()

        doc = self.nlp(text.lower())

       
        emotion_scores = []
        detected_emotions = {}

        for emotion, data in self.emotion_keywords.items():
            intensity = self._calculate_intensity(text.lower(), data['keywords'])
            if intensity > 0:
                emotion_scores.append(intensity * data['polarity'])
                detected_emotions[emotion] = intensity

        sentiment_score = np.mean(emotion_scores) if emotion_scores else 0

        results = {
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'text': text,
            'mood_score': sentiment_score,
            'overall_mood': 'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral',
            'emotions': detected_emotions
        }

        self.history.append(results)
        return results

    def _calculate_intensity(self, text, keywords):
        count = sum(1 for keyword in keywords if keyword in text)
        return min(count * 3.5, 10)

    def format_output(self, analysis):
        output = []
        output.append("=== Emotional Analysis ===")
        output.append(f"Time: {analysis['timestamp']}")
        output.append(f"Your response: {analysis['text']}")
        output.append("\nMood Analysis:")
        output.append(f"  Overall Mood: {analysis['overall_mood']}")
        output.append(f"  Mood Score: {analysis['mood_score']:.2f}")

        output.append("\nDetected emotions:")
        for emotion, intensity in analysis['emotions'].items():
            polarity = self.emotion_keywords[emotion]['polarity']
            polarity_label = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
            output.append(f"  • {emotion}: Intensity {intensity:.1f}/10 ({polarity_label})")

        return "\n".join(output)

    def analyze_trends(self):
        if not self.history:
            return "No historical data available for trend analysis."

        try:
            df_data = []
            for entry in self.history:
                row = {
                    'timestamp': datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S"),
                    'mood_score': entry['mood_score']
                }
                for emotion in self.emotion_keywords.keys():
                    row[emotion] = entry['emotions'].get(emotion, 0)
                df_data.append(row)

            df = pd.DataFrame(df_data)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

            df.plot(x='timestamp', y='mood_score', marker='o', ax=ax1)
            ax1.set_title('Mood Score Over Time')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Mood Score')
            ax1.grid(True)

            emotions_data = df[[emotion for emotion in self.emotion_keywords.keys()]].mean().sort_values(ascending=False)
            emotions_data = emotions_data[emotions_data > 0]  # Only show emotions that were detected

            if not emotions_data.empty:
                sns.barplot(x=emotions_data.values, y=emotions_data.index, ax=ax2)
                ax2.set_title('Average Emotion Intensities')
                ax2.set_xlabel('Intensity')

            plt.tight_layout()
            plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'trends.png'))
            plt.close()

            stats = self._calculate_statistics(df)

            return stats

        except Exception as e:
            return f"Error during trend analysis: {str(e)}\nPlease ensure you have entered multiple responses before viewing trends."

    def _calculate_statistics(self, df):
        stats = []
        stats.append("\n=== Trend Analysis ===")

        avg_mood = df['mood_score'].mean()
        stats.append(f"\nOverall Mood Statistics:")
        stats.append(f"Average Mood Score: {avg_mood:.2f}")
        stats.append(f"Mood Variance: {df['mood_score'].var():.2f}")

        stats.append("\nEmotion Frequency Analysis:")
        emotion_means = df[[emotion for emotion in self.emotion_keywords.keys()]].mean()
        detected_emotions = emotion_means[emotion_means > 0].sort_values(ascending=False)

        for emotion in detected_emotions.index:
            polarity = self.emotion_keywords[emotion]['polarity']
            polarity_label = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
            avg_intensity = detected_emotions[emotion]
            stats.append(f"{emotion} ({polarity_label}): Average Intensity {avg_intensity:.1f}/10")

        if not detected_emotions.empty:
            dominant_emotion = detected_emotions.index[0]
            stats.append(f"\nDominant Emotion: {dominant_emotion}")

        return "\n".join(stats)

analyzer = EmotionalAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    analysis = analyzer.analyze_text(text)
    return render_template('analysis.html', analysis=analysis)

@app.route('/trends', methods=['GET'])
def trends():
    stats = analyzer.analyze_trends()
    return render_template('trends.html', stats=stats)

if __name__ == "__main__":
    app.run(debug=True)