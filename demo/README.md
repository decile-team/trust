## Overview

In this demo, we demonstrate an active learning based tool that can facilitate selection of iterative subsets in any real-world scenario. We show the demo for class imbalance and out-of-distribution scenarios. Our tool makes it easy for user to change parameters and acquisition functions in-order to iterate over the subset selection process quickly. Concretely, the user can visualize the selected subset to get a high level idea if the selected subset satisfies the criteria they are looking for. This criteria can be diversity or query-relevance and may be specific to the real-world scenario. For instance, in class imbalance scenarios, they user will try to select data points from the rare classes. Once the use is satisfies with the selected subset, they can export it for labeling and send it to a team of human annotators. This iterative process of subset selection and visualization helps ensures that the best representative subset is sent for labeling and the best use of labeling costs is made. 

Please watch the "How to use" video below to get a taste of our demo in action!

## Virtual Environment

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

## Requirements

```
pip install -r requirements.txt
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ submodlib
```

## Run

Open constants.py and change the path of demo video(demo_video_path) to your local repository.

Finally run the app using the following command:

```
python3 app.py
```

## How to use (Video)

https://user-images.githubusercontent.com/38362966/209561934-73514537-f4ce-490f-a5a3-2be59910ea2b.mov


The app will open up in port 9000, open it in your browser.
You can find a "How to use?" button in the app, which will guide you through the process.
