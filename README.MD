# Automation library for ADB / Android

## pip install usefuladbplus

### Tested against Windows 10 / Python 3.11 / Anaconda / BlueStacks5

## If you get an Exception (deepcopy .. blah blah ...), then **pip install pandas==2.1.4**

This package is designed for automating interactions with Android devices. Whether you're developing mobile applications, performing UI testing, or exploring Android internals, this toolkit provides an extensive set of functionalities to streamline your workflow.

## Key Features:



### Fast Screenshot Capture: 

Capture device screens at high speed using native tools, providing quick access to visual information.

### UI Exploration: 

Dive deep into the Android UI hierarchy, extracting valuable data such as element properties, text, and screen coordinates.

### Interaction Automation: 

Automate device interactions, including tapping on elements, sending events, and more, allowing for efficient UI testing and exploration.

### Color Analysis: 

Employ advanced color search algorithms to identify specific colors within screenshots, enabling precise element identification.

### Shape and Cluster Recognition: 

Detect and interact with shapes and color clusters, facilitating complex automation scenarios.

### Template Matching: 

Utilize template matching techniques for recognizing predefined image patterns on the device screen.

### Text Recognition with OCR: 

Leverage Tesseract OCR to extract text from screenshots, enabling text-based automation and validation.

### Fuzzy String Matching: 

Perform fuzzy matching between strings, enhancing the accuracy of string-based comparisons in automation workflows.

### Input Device Recording and Replay: 

Record device input events, save them for future use, and replay them to automate repetitive tasks.

### Includes all methods from 

https://github.com/hansalemaos/usefuladb

## Activating Installed Modules:

```python
import cv2
from usefuladbplus import activate_pandas_extensions, AdbControlPlus

# If the pandas DataFrames get printed slowly, consider using PrettyColorPrinter
from PrettyColorPrinter import add_printer
add_printer(1)


# Activates custom extensions
# Each module corresponds to a specific extension and must be installed separately using pip.
# plus_find_shapes - pip install multiprocshapefinder
# plus_template_matching - pip install needlefinder
# plus_color_search_c - pip install chopchopcolorc
# plus_color_cluster - pip install locatecolorcluster
# plus_count_all_colors - pip install colorcountcython
# plus_count_all_colors_coords - pip install colorcountcython
# plus_fuzzy_merge - pip install a_pandas_ex_fuzzymerge
# plus_tesser_act - pip install multitessiocr
activate_pandas_extensions(modules=(
        "plus_find_shapes",
        "plus_template_matching",
        "plus_color_search_c",
        "plus_color_cluster",
        "plus_count_all_colors",
        "plus_count_all_colors_coords",
        "plus_fuzzy_merge",
        "plus_tesser_act",
    ))

```
## ADB Connection Setup:


```python
addr = "127.0.0.1:5695"
adb_path = r"C:\Android\android-sdk\platform-tools\adb.exe"
# To connect to all devices at once, you can use this static method (Windows only):
AdbControlPlus.connect_to_all_tcp_devices_windows(
    adb_path=adb_path,
)
```

## Creating an instance 

```python
adb = AdbControlPlus(
    adb_path=adb_path,
    device_serial=addr,
    use_busybox=False,
    connect_to_device=True,
    invisible=True,
    print_stdout=True,
    print_stderr=True,
    limit_stdout=3,
    limit_stderr=3,  
    limit_stdin=None,
    convert_to_83=True,
    wait_to_complete=0.1,
    flush_stdout_before=True,
    flush_stdin_before=True,
    flush_stderr_before=True,
    exitcommand="xxxCOMMANDxxxDONExxx",
    capture_stdout_stderr_first=True,
    global_cmd=False,
    global_cmd_timeout=10,
    use_eval=True,  # executes commands using eval - which is recommended - since it is the fastest version - check out https://github.com/hansalemaos/usefuladb for more information about the connection
    eval_timeout=180,  # timeout for eval (stdout/stderr netcat data transfer)
)

```

## UiAutomator Dump / Activity (Fragment Manager) Dump 

```python
# Starting point is always a dump from either the activities or uiautomator

# Advantages of the Activities Dump:
# - Very fast
# - Reliable, rarely fails
# - Includes information that UiAutomator doesn't have

# Disadvantages:
# - Generally provides less information than UiAutomator
# - Lacks content/text capture functionality
# - Elements from background processes might disturb the accuracy of the dump

df = adb.plus_activity_elements_dump(
    with_screenshot=True,
    screenshot=None,
    with_sendevent=False,
)
# The resulting dataframe should look like this:

#    aa_start_x  aa_start_y  aa_center_x  aa_center_y  aa_area  aa_end_x  aa_end_y  aa_width  aa_height  aa_start_x_relative  aa_start_y_relative  aa_end_x_relative  aa_end_y_relative      aa_coords       aa_int_coords              aa_classname aa_hashcode           aa_element_id   aa_mid aa_visibility aa_focusable aa_enabled aa_drawn aa_scrollbars_horizontal aa_scrollbars_vertical aa_clickable aa_long_clickable aa_context_clickable aa_pflag_is_root_namespace aa_pflag_focused aa_pflag_selected aa_pflag_prepressed aa_pflag_hovered aa_pflag_activated aa_pflag_invalidated aa_pflag_dirty_mask  aa_is_parent  aa_view_index                           aa_aa_parents  aa_element_index aa_screenshot ff_input_tap ff_sendevent_event4 ff_sendevent_event5
# 0         343         577          352          586      324       361       595        18         18                   60                    2                 78                 20     60,2-78,20     (60, 2, 78, 20)  android.widget.ImageView   #7f08011b  app:id/popup_image_one  fb7739e             V            .          E        D                        .                      .            .                 .                    .                          .                .                 .                   .                .                  .                    I                   D          True              0  (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)                 0    [[[255 255     352, 586            352, 586            352, 586
# 1         260         661          323          680     4953       387       700       127         39                    0                  113                127                152  0,113-127,152  (0, 113, 127, 152)   android.widget.TextView   #7f080060     app:id/app_name_one   8a8e7f             V            .          E        D                        .                      .            .                 .                    .                          .                .                 .                   .                .                  .                    .                   .          True              0      (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)                 1    [[[255 255     323, 680            323, 680            323, 680
# 2         470         577          479          586      324       488       595        18         18                   60                    2                 78                 20     60,2-78,20     (60, 2, 78, 20)  android.widget.ImageView   #7f08011e  app:id/popup_image_two  da32baa             I            .          E        D                        .                      .            .                 .                    .                          .                .                 .                   .                .                  .                    I                   D          True              0  (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15)                 2    [[[255 255     479, 586            479, 586            479, 586


# Advantages of the UiAutomator Dump:
# - Text capture
# - Large quantity of items
# - Very accurate

# Disadvantages:
# - May fail if there is too much activity on the screen
# - Sometimes takes a considerable amount of time

df = adb.plus_uidump(
    timeout=60,
    with_screenshot=True,
    screenshot=None,
    nice=False,
    su=False,
    with_sendevent=True,
)

# This method addresses the timeout issue in UiAutomator when there is too much activity during XML dumping.
# It freezes the PID (Process ID) before launching UiAutomator.
# Ensure that you block the correct PID; blocking the wrong one may cause the process to freeze completely,
# resulting in UiAutomator being unable to dump anything.
# Here is an example of performing a dump with the Kiwi Browser using this approach.

df = adb.plus_uidump_with_freeze(
    procregex_for_lsof=rb"^\s*dboxed_process.*kiwi",
    timeout=15,
    with_screenshot=True,
    screenshot=None,
    with_sendevent=True,
)

# All columns starting with 'ff' represent functions.
# You can filter the object you want to interact with using the following example:

filtered_df = df.loc[df.aa_element_id.str.contains('refresh_.*', na=False, regex=True)]

# Clicking is as simple as:

filtered_df.ff_input_tap.iloc[0]()

# An example of using 'sendevent' (only possible with SU privileges):

df_with_sendevent = adb.plus_activity_elements_dump(
    with_screenshot=True,
    screenshot=None,
    with_sendevent=True,
)

# Performing a 'sendevent' action on the filtered element:

df_with_sendevent.loc[filtered_df.index[0]].ff_sendevent_event4.iloc[0]()

# Screenshots can be saved to a folder using the following command:
# The folder will be created if it does not exist.

# Example:
df.plus_save_screenshots("C:\\dumpscreenshotstest2")


```


# Finding the right element 


## RGB COLOR SEARCH

```python
# Uses a very fast algorithm written in C that searches for RGB colors in images.
dfcolors = df.plus_color_search_c(
    colors2find=(
        (26, 115, 232),
        (70, 70, 73),
    ),
    column="aa_screenshot",
    cpus=5,
    chunks=1,
    print_stderr=True,
    print_stdout=False,
    usecache=True,
    with_sendevent=True
)

# It returns a copy of the DataFrame with at least 4 new columns (more than 4 if with_sendevent is True):
#   aa_tag aa_index aa_text aa_resource-id                     aa_class               aa_package aa_content-desc aa_checkable aa_checked aa_clickable aa_enabled aa_focusable aa_focused aa_scrollable aa_long-clickable aa_password aa_selected           aa_bounds aa_rotation aa_all_parents aa_all_children aa_start_x aa_start_y aa_end_x aa_end_y aa_area aa_center_x aa_center_y aa_width aa_height  aa_ratio                  aa_screenshot ff_input_tap ff_sendevent_event4 ff_sendevent_event5               aa_colorsearch_c  aa_csearch_mean_x  aa_csearch_mean_y ff_colormean_input_tap ff_colormean_sendevent_event4 ff_colormean_sendevent_event5
# 2   node        0                          android.widget.FrameLayout  com.kiwibrowser.browser                        False      False        False       True        False      False         False             False       False       False   (0, 0, 1280, 720)           0             ()            (4,)          0          0     1280      720  921600         640         360     1280       720  1.777778  [[[253, 249, 247], [253, 2...     640, 360            640, 360            640, 360  [[534, 872], [534, 873], [...              903.0              549.0               903, 549                      903, 549                      903, 549
# 3   node        0                         android.widget.LinearLayout  com.kiwibrowser.browser                        False      False        False       True        False      False         False             False       False       False   (0, 0, 1280, 720)           0             ()            (5,)          0          0     1280      720  921600         640         360     1280       720  1.777778  [[[253, 249, 247], [253, 2...     640, 360            640, 360            640, 360  [[534, 872], [534, 873], [...              903.0              549.0               903, 549                      903, 549                      903, 549
# 4   node        0                          android.widget.FrameLayout  com.kiwibrowser.browser                        False      False        False       True        False      False         False             False       False       False  (0, 24, 1280, 720)           0           (2,)            (6,)          0         24     1280      720  890880         640         372     1280       696   1.83908  [[[250, 243, 239], [250, 2...     640, 372            640, 372            640, 372  [[534, 872], [534, 873], [...              903.0              573.0               903, 573                      903, 573                      903, 573


# If you want to apply the search only to some elements (usually it is not necessary to filter all elements in the whole dataframe),
# you can use slicing:

# Apply color search to the first 10 elements of the DataFrame.


dfcolors = df[:10].plus_color_search_c(
    colors2find=(
        (26, 115, 232),
        # (70, 70, 73),
    ),
    column="aa_screenshot",
    cpus=5,
    chunks=1,
    print_stderr=True,
    print_stdout=False,
    usecache=True,
)
```

## CLUSTER COLOR SEARCH

```python
# You can also cluster colors if clicking on the mean value is too risky for you:


# Perform color clustering on the DataFrame.
clustercordf = df.plus_color_cluster(
    colors=((26, 115, 232),),
    reverse_colors=True,  # input is BGR, but you can reverse the colors
    backend="C",  # C is much faster than scipy, but if the array gets too big for your memory, the process crashes
    memorylimit_mb=10000,
    eps=3,
    min_samples=10,
    algorithm="auto",
    leaf_size=30,
    n_jobs=5,
    max_width=100,
    max_height=100,
    interpolation=cv2.INTER_NEAREST,
    # cv2.INTER_NEAREST is the best here, since it doesn't change the color, it is better not to use cv2.INTER_LANCZOS4
    column="aa_screenshot",
    with_sendevent=True,
)

# You can click on a cluster like this:
clustercordf.ff_cluster_input_tap.iloc[3]()
# or with sendevent (SU!!)
clustercordf.ff_cluster_sendevent_event4.iloc[0]()

# Results can be seen like this:
clustercordf.plus_save_screenshots(r'C:\vsvfddds')

```

## SHAPE SEARCH


```python
# You can also search for shapes:

# Sort the DataFrame by "aa_area" in descending order, select a subset, and perform shape finding.
df4 = (
    df.sort_values(by="aa_area", ascending=False)
    .iloc[12:20]
    .plus_find_shapes(
        with_draw_function=True,
        threshold1=10,
        threshold2=90,
        approxPolyDPvar=0.01,
        cpus=5,
        chunks=1,
        print_stderr=True,
        print_stdout=False,
        usecache=True,
        with_sendevent=True
    )
)

# The resulting DataFrame contains information about the found shapes.

# Results can be saved to HDD and checked like this:
# You need to pass with_draw_function=True when you call plus_find_shapes

# Add a column "aa_screenshot" with the drawn shapes.
df4['aa_screenshot'] = df4.ff_drawn_shape.apply(lambda x: x())
# or
# Save the screenshots to a folder.
df4.plus_save_screenshots('c:\\shapescs')

```

## TEMPLATE MATCHING

```python
# Define a dictionary with needle images and their file paths.
needles = {
    'b1': r"C:\scrfadx\36.png",
    'b2': r"C:\scrfadx\42.png",
    'b3': r"C:\scrfadx\45.png",
    'b4': r"C:\scrfadx\48.png",
}

# Apply template matching on the DataFrame using the defined needle images.
dftemp = df.plus_template_matching(
    needles,
    with_sendevent=True,
    with_image_data=True,
    thresh=0.9,
    pad_input=False,
    mode="constant",
    constant_values=0,
    usecache=True,
    processes=5,
    chunks=1,
    print_stdout=False,
    print_stderr=True,
    column="aa_screenshot",
)

# The resulting DataFrame contains information about the matched templates.

#   aa_tag aa_index aa_text aa_resource-id                    aa_class               aa_package aa_content-desc aa_checkable aa_checked aa_clickable aa_enabled aa_focusable aa_focused aa_scrollable aa_long-clickable aa_password aa_selected          aa_bounds aa_rotation aa_all_parents aa_all_children aa_start_x aa_start_y aa_end_x aa_end_y aa_area aa_center_x aa_center_y aa_width aa_height  aa_ratio                  aa_screenshot ff_input_tap ff_sendevent_event4 ff_sendevent_event5  aa_realindex  aa_needle_abs_start_x  aa_needle_abs_start_y  aa_needle_abs_end_x  aa_needle_abs_end_y  aa_needle_abs_center_x  aa_needle_abs_center_y  aa_img_index  aa_needle_start_x  aa_needle_start_y  aa_needle_scale_factor  aa_needle_width  aa_needle_height  aa_needle_match  aa_needle_end_x  aa_needle_end_y  aa_needle_center_x  aa_needle_center_y  aa_needle_area aa_needle_needlename  aa_needle_img_index           aa_needle_screenshot  aa_needle_r  aa_needle_g  aa_needle_b ff_needle_input_tap ff_needle_sendevent_event4 ff_needle_sendevent_event5
# 0   node        0                         android.widget.FrameLayout  com.kiwibrowser.browser                        False      False        False       True        False      False         False             False       False       False  (0, 0, 1280, 720)           0             ()            (4,)          0          0     1280      720  921600         640         360     1280       720  1.777778  [[[253, 249, 247], [253, 2...     640, 360            640, 360            640, 360             2                    866                    557                  939                  589                   902.5                     573             2                866                557                     100               73                32              1.0              939              589               902.5               573.0            2336                   b1                    0  [[[255, 255, 255], [255, 2...    54.235873   131.822346   234.391267        902.5, 573.0               902.5, 573.0               902.5, 573.0
# 1   node        0                         android.widget.FrameLayout  com.kiwibrowser.browser                        False      False        False       True        False      False         False             False       False       False  (0, 0, 1280, 720)           0             ()            (4,)          0          0     1280      720  921600         640         360     1280       720  1.777778  [[[253, 249, 247], [253, 2...     640, 360            640, 360            640, 360             2                    143                     63                  191                  119                   167.0                      91             2                143                 63                     100               48                56              1.0              191              119               167.0                91.0            2688                   b3                    0  [[[250, 243, 239], [250, 2...    54.235873   131.822346   234.391267         167.0, 91.0                167.0, 91.0                167.0, 91.0
# 2   node        0                         android.widget.FrameLayout  com.kiwibrowser.browser                        False      False        False       True        False      False         False             False       False       False  (0, 0, 1280, 720)           0             ()            (4,)          0          0     1280      720  921600         640         360     1280       720  1.777778  [[[253, 249, 247], [253, 2...     640, 360            640, 360            640, 360             2                    191                     67                  239                  115                   215.0                      91             2                191                 67                     100               48                48              1.0              239              115               215.0                91.0            2304                   b4                    0  [[[255, 255, 255], [255, 2...    54.235873   131.822346   234.391267         215.0, 91.0                215.0, 91.0                215.0, 91.0

# Clicking on an element:
# Execute the sendevent method for the first matched needle element.
dftemp.ff_needle_sendevent_event4.iloc[0]()

```

## COLOR COUNT 

```python
# Counts all colors (no clicking methods, because the frame is too big)
# Count the occurrence of each color in the screenshots.
dfallcolors = df.plus_count_all_colors(column="aa_screenshot")
```

## COLOR COUNT - coordinates

```python
# Returns all coordinates for each color (no clicking methods, because the frame is too big)
# Count and retrieve the coordinates for each color in the screenshots.
dfallcolorscount = df.plus_count_all_colors_coords(column="aa_screenshot")
```

## OCR

```python
# OCR on images with tesseract
# Apply OCR (Optical Character Recognition) using Tesseract on the screenshots.
dftes = df.plus_tesser_act(
    column="aa_screenshot",
    tesser_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    add_after_tesseract_path="",
    add_at_the_end="-l eng+por --psm 3",
    processes=5,
    chunks=1,
    print_stdout=False,
    print_stderr=False,
)

# Perform a click action if the recognized text is "Bingo."
dftes.loc[dftes.tt_text == "Bingo"].tt_sendevent_event4.iloc[0]()
```

## FUZZY MATCH

```python
import string
import random
from rapidfuzz import fuzz

# Dummy data
# Generate a list of strings for testing.
stringlisttest = [
    str(x).replace(random.choice(x), random.choice(string.ascii_letters))
    for x in df['aa_class'].unique()
]

# Perform fuzzy matching between the generated list and the 'aa_class' column in the DataFrame.
dffuzzy = df.plus_fuzzy_merge(
    stringlisttest, column="aa_class", scorer=fuzz.WRatio, min_value=0
)

# Print the resulting DataFrame with selected columns for demonstration.
# print(dffuzzy[['aa_class_x', 'aa_class_y', 'concat_value']][:5].to_string())
#                     aa_class_x                   aa_class_y  concat_value
# 0   android.widget.FrameLayout   anWroiW.wiWget.FrameLayout            88
# 1  android.widget.LinearLayout  aedroid.widget.LieearLayout            93
# 2   android.widget.FrameLayout   anWroiW.wiWget.FrameLayout            88
# 3   android.widget.FrameLayout   anWroiW.wiWget.FrameLayout            88
# 4   android.widget.FrameLayout   anWroiW.wiWget.FrameLayout            88

```

## RECORD / REPLAY ACTIONS

```python
# How to record and replay actions (requires SU!)

# Get all input devices
# List all available input devices, including event files.
adb.sh_list_dev_input()
# ['/dev/input/event0',
#  '/dev/input/event1',
#  '/dev/input/event2',
#  '/dev/input/event3',
#  '/dev/input/event4',
#  '/dev/input/event5',
#  '/dev/input/event6',
#  '/dev/input/mice']

# Record and save events to a specified folder on the SD card.
event = adb.plus_record_and_save_to_sdcard(
    tmpfolder_device="/sdcard/clickevent21",
    tmpfolder_local=None,
    device="event4",
    print_output=True,
    add_closing_command=True,
    clusterevents=16,
)
# event
# Out[18]: su -c 'cat /sdcard/clickevent21/cmd.txt | sh'

# The recorded events are now saved on the SD card and can be executed as follows:

# Execute the recorded events using the saved command.
adb.execute_sh_command(event.execution)
# or
# Execute the recorded events using the command directly.
adb.execute_sh_command("""su -c 'cat /sdcard/clickevent21/cmd.txt | sh'""")
```

## FAST SCREENSHOTS 

```python
# Capture screen frames using screenrecord, convert them to numpy arrays, and display using OpenCV

# Iterate through frames captured with fast screenshot method
for image in adb.plus_start_fast_screenshot_iter(
    bitrate="20M",
    screenshotbuffer=10,
    go_idle=0,
):
    # Display the captured frame using OpenCV
    cv2.imshow("CV2 WINDOW", image)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close OpenCV windows when the loop is exited
cv2.destroyAllWindows()

```

