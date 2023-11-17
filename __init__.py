import os
import re
import subprocess
import sys
import time
from collections import defaultdict
import cv2
import numexpr
from a_cv_imwrite_imread_plus import save_cv_image
from adbeventparser import EventRecord
from adbnativeblitz import AdbFastScreenshots
from flatten_everything import flatten_everything
from geteventplayback import GeteventPlayBack
from istruthy import is_truthy
from touchtouch import touch
from usefuladb import AdbControl
import pandas as pd
import numpy as np
from a_pandas_ex_apply_ignore_exceptions import pd_add_apply_ignore_exceptions

pd_add_apply_ignore_exceptions()
adbconfig = sys.modules[__name__]
adbconfig.all_devices = {}


class RecordedEvent:
    """
    Represents a recorded event with an associated command for execution.

    Attributes:
    - fu: The function for executing the recorded event.
    - command: The command used for recording the event.
    - execution: The full execution command for replaying the recorded event.


    """

    def __init__(self, fu, command):
        self.command = command
        self.execution = f"su -c 'cat {command} | sh'"
        self.fu = fu

    def __call__(self, **kwargs):
        self.fu(self.execution, **kwargs)

    def __str__(self):
        return self.execution

    def __repr__(self):
        return self.__str__()


class AdbDescriptor:
    """
    Solution for inheriting attributes from a Pandas DataFrame
    Descriptor for accessing the AdbControl instance associated with a class instance.

    """

    def __get__(self, instance, owner):
        try:
            if not instance.__dict__[self.name]:
                instance.__dict__[self.name] = instance.__dict__["_attrs"][
                    "adb_instance"
                ]

        except Exception:
            instance.__dict__[self.name] = None
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        try:
            instance.__dict__[self.name] = instance.__dict__["_attrs"]["adb_instance"]
        except Exception:
            instance.__dict__[self.name] = value

    def __delete__(self, instance):
        return

    def __set_name__(self, owner, name):
        self.name = name


class DataFrameWithMeta(pd.DataFrame):
    """
    Subclass of pandas DataFrame with additional metadata, specifically designed for ADB interactions.

    Attributes:
    - adb_instance: A descriptor attribute providing access to the associated AdbControl instance.

    Example:
    ```
    adb_plus = AdbControlPlus(adb_path="/path/to/adb", device_serial="123456")
    df = DataFrameWithMeta(data=my_data, adb_instance=adb_plus)
    df.adb_instance  # Access the associated AdbControlPlus instance
    ```

    """

    adb_instance = AdbDescriptor()

    @property
    def _constructor(self):
        return self.__class__

    def __init__(self, *args, adb_instance=None, **kwargs):
        self.adb_instance = adb_instance

        super().__init__(*args, **kwargs)
        self.attrs["adb_instance"] = adb_instance

    def plus_save_screenshots(self, folder, column="aa_screenshot"):
        r"""
        Save screenshots from the specified column to the given folder.

        Args:
        - folder (str): The folder path where the screenshots will be saved.
        - column (str, optional): The column containing screenshots. Defaults to "aa_screenshot".

        Returns:
        pandas.Series: A Series with file paths of the saved screenshots.

        Example:
        saved_files = df.plus_save_screenshots(folder="/path/to/screenshots")

        """
        return self.ds_apply_ignore(
            pd.NA,
            lambda x: save_cv_image(
                f'{os.path.join(folder, str(x.name) + ".png")}', x[column]
            )
            if x.aa_area > 0
            else pd.NA,
            axis=1,
        )


class DataFrameWithMetaShaply(DataFrameWithMeta):
    r"""
    Subclass of DataFrameWithMeta with additional functionality for drawing and saving Shapely polygons on images.

    Attributes:
    - color (tuple, optional): RGB color tuple for drawing polygons. Defaults to (255, 0, 255).
    - thickness (int, optional): Thickness of polygon edges. Defaults to 2.


    """

    def plus_save_screenshots(
        self, folder, column="aa_screenshot", color=(255, 0, 255), thickness=2
    ):
        """
        Save screenshots with Shapely polygons drawn on them to the specified folder.

        Args:
        - folder (str): The folder path where the screenshots will be saved.
        - column (str, optional): The column containing screenshots. Defaults to "aa_screenshot".
        - color (tuple, optional): RGB color tuple for drawing polygons. Defaults to (255, 0, 255).
        - thickness (int, optional): Thickness of polygon edges. Defaults to 2.

        Returns:
        pandas.Series: A Series with file paths of the saved screenshots.

        Example:
        saved_files = df_shaply.plus_save_screenshots(folder="/path/to/screenshots")

        """
        allfi = []
        for i, va in self.iterrows():
            try:
                im = va[column].copy()
                ff = va.boundary

                cv2.fillPoly(im, [ff], color)
                cv2.polylines(im, [ff], isClosed=True, color=color, thickness=thickness)
                filep = os.path.join(folder, str(i) + ".png")
                save_cv_image(f"{filep}", im)
                allfi.append(filep)
            except Exception:
                allfi.append(pd.NA)

        return pd.Series(allfi, index=self.index.copy())


class DrawExecutor:
    r"""
    Executor class for drawing and saving shapes on images based on provided function and DataFrame.

    Attributes:
    - fu: The drawing function.
    - frame: The DataFrame containing information about shapes.
    - screenshot: The screenshot image.

    """

    def __init__(self, fu, frame, screenshot):
        self.fu = fu
        self.screenshot = screenshot
        self.frame = frame

    def __call__(
        self,
        save_folder=None,
        min_area=100,
        shapes=("rectangle", "triangle", "circle", "pentagon", "hexagon", "oval"),
    ):
        try:
            if min_area > self.frame.aa_area.iloc[0]:
                return pd.NA
            sho = self.fu(
                self.frame,
                self.screenshot,
                min_area=min_area,
                shapes=shapes,
                cv2show=False,
            )
            if save_folder:
                save_cv_image(
                    os.path.join(save_folder, str(self.frame.index[0]) + ".png"), sho
                )
            return sho
        except Exception as fe:
            sys.stderr.write(f"{fe}\n")
            sys.stderr.flush()
            return pd.NA

    def __str__(self):
        return "()"

    def __repr__(self):
        return "()"


def _pandas_ex_fuzzy_match(df, stringlist, column=None, scorer=None, min_value=0):
    r"""
    Perform fuzzy matching between a DataFrame column and a list of strings.

    Parameters:
    - df (DataFrameWithMeta): The input DataFrame containing the data.
    - stringlist (list): The list of strings to match against the DataFrame column.
    - column (str, optional): The column in the DataFrame to perform the fuzzy matching on.
      Defaults to 'aa_text' if not specified.
    - scorer (callable, optional): The scoring function for fuzzy matching.
      Defaults to fuzz.WRatio from the rapidfuzz library if not specified.
    - min_value (int, optional): The minimum matching score for a match to be considered.
      Matches with scores below this threshold will be excluded from the result.
      Defaults to 0 if not specified.

    Returns:
    - DataFrameWithMeta: A new DataFrame containing rows that match the specified criteria.

    Example:
    ```
    stringlist = ['Pôxxquer', 'Quxxs', 'Quênixxa - Premier League', ... ]
    df3 = df.plus_fuzzy_match(stringlist=stringlist, column='aa_text', scorer=fuzz.WRatio, min_value=0)
    print(df3)
    ```

    Note:
    - This function uses the rapidfuzz library (https://pypi.org/project/rapidfuzz/ - written in C++) for fuzzy matching.

    """

    if not scorer:
        scorer = fuzz.WRatio
    if not column:
        column = "aa_text"

    df1 = pd.DataFrame(stringlist, columns=[column])
    df3 = df.d_fuzzy_merge(
        df1,
        right_on=column,
        left_on=column,
        usedtype=np.uint8,
        scorer=scorer,
        concat_value=True,
    )
    return DataFrameWithMeta(
        df3.loc[df3.concat_value > min_value],
        adb_instance=df.adb_instance,
    )


def _pandas_ex_count_colors(
    df,
    column="aa_screenshot",
):
    r"""
    Count the occurrence of colors in images stored in a DataFrame column.

    Parameters:
    - df (DataFrameWithMeta): The input DataFrame containing the data.
    - column (str, optional): The column in the DataFrame containing images.
      Defaults to 'aa_screenshot' if not specified.

    Returns:
    - DataFrameWithMeta: A new DataFrame containing color count information.

    Example:
    ```
    dft = df.plus_count_colors(column='aa_screenshot')
    print(dft)
    ```

    Note:
    - This function uses the `colorcount` (https://github.com/hansalemaos/colorcountcython - written in Cython) function to count the occurrence of colors in each image.
    - The resulting DataFrame includes columns 'aa_img_index', 'aa_r', 'aa_g', 'aa_b', and 'aa_count'.
    - 'aa_img_index' represents the index of the image in the DataFrame.
    - 'aa_r', 'aa_g', and 'aa_b' represent the red, green, and blue components of the color.
    - 'aa_count' represents the count of the corresponding color in the image.
    """
    dft = (
        df[column]
        .ds_apply_ignore(
            ((), ()),
            lambda b: colorcount(pic=b, coords=False, count=True)[
                "color_count"
            ].items(),
        )
        .to_frame()
        .explode(column)
        .dropna()
        .ds_apply_ignore(
            [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
            lambda q: [
                q.name,
                q[column][0][0],
                q[column][0][1],
                q[column][0][2],
                q[column][1],
            ],
            result_type="expand",
            axis=1,
        )
        .dropna()
        .astype({0: np.uint32, 1: np.uint8, 2: np.uint8, 3: np.uint8, 4: np.uint32})
        .rename(
            columns={0: "aa_img_index", 1: "aa_r", 2: "aa_g", 3: "aa_b", 4: "aa_count"}
        )
    )
    return DataFrameWithMeta(
        dft,
        adb_instance=df.adb_instance,
    )


def _pandas_ex_color_coords(
    df,
    column="aa_screenshot",
):
    r"""
    Extract color coordinates from images stored in a DataFrame column.

    Parameters:
    - df (DataFrameWithMeta): The input DataFrame containing the data.
    - column (str, optional): The column in the DataFrame containing images.
      Defaults to 'aa_screenshot' if not specified.

    Returns:
    - DataFrameWithMeta: A new DataFrame containing color coordinate information.

    Example:
    ```
    df_color_coords = df.plus_color_coords(column='aa_screenshot')
    print(df_color_coords)
    ```

    Note:
    - This function uses the `colorcount` (https://github.com/hansalemaos/colorcountcython - written in Cython) function to extract color coordinates from each image.
    - The resulting DataFrame includes columns 'aa_img_index', 'aa_r', 'aa_g', 'aa_b', 'aa_x', and 'aa_y'.
    - 'aa_img_index' represents the index of the image in the DataFrame.
    - 'aa_r', 'aa_g', and 'aa_b' represent the red, green, and blue components of the color.
    - 'aa_x' and 'aa_y' represent the x and y coordinates of the color in the image.
    """
    dfcolorcoords = df[column].ds_apply_ignore(
        pd.NA,
        lambda b: colorcount(pic=b, coords=True, count=False)["color_coords"].items(),
    )
    dfcolorcoords.to_frame().dropna().explode(column).apply(
        lambda q: q[column][0], result_type="expand", axis=1
    )
    dfcolorcoords = (
        dfcolorcoords.to_frame()
        .dropna()
        .explode(column)
        .apply(
            lambda q: [
                q.name,
                q[column][0][0],
                q[column][0][1],
                q[column][0][2],
                q[column][1],
            ],
            result_type="expand",
            axis=1,
        )
        .explode(4)
        .dropna()
        .astype({0: np.uint32, 1: np.uint8, 2: np.uint8, 3: np.uint8})
    )
    dfcolorcoords["aa_x"] = dfcolorcoords[4].str[0].astype(np.uint32)
    dfcolorcoords["aa_y"] = dfcolorcoords[4].str[1].astype(np.uint32)
    dfcolorcoords.drop(columns=4, inplace=True)
    dfcolorcoords.reset_index(drop=True, inplace=True)
    dfcolorcoords.rename(
        columns={0: "aa_img_index", 1: "aa_r", 2: "aa_g", 3: "aa_b"}, inplace=True
    )
    return DataFrameWithMeta(
        dfcolorcoords,
        adb_instance=df.adb_instance,
    )


def _pandas_ex_color_search_with_c(
    df,
    colors2find,
    column="aa_screenshot",
    cpus=5,
    chunks=1,
    print_stderr=True,
    print_stdout=False,
    usecache=True,
    with_sendevent=False,
):
    r"""
    Search for specified colors in images stored in a DataFrame column using C-extension.

    Parameters:
    - df (DataFrameWithMeta): The input DataFrame containing the data.
    - colors2find (list): A list of RGB tuples representing colors to search for in the images.
    - column (str, optional): The column in the DataFrame containing images.
      Defaults to 'aa_screenshot' if not specified.
    - cpus (int, optional): The number of CPU cores to use for parallel processing. Defaults to 5.
    - chunks (int, optional): The number of chunks to divide the data into for parallel processing. Defaults to 1.
    - print_stderr (bool, optional): Whether to print stderr messages during color search. Defaults to True.
    - print_stdout (bool, optional): Whether to print stdout messages during color search. Defaults to False.
    - usecache (bool, optional): Whether to use caching for color search results. Defaults to True.
    - with_sendevent (bool, optional): Whether to include sendevent information in the result. Defaults to False.

    Returns:
    - DataFrameWithMeta: The input DataFrame with additional columns related to color search.

    Example:
    ```
    df = df.plus_color_search_with_c(
        colors2find=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        column='aa_screenshot',
        cpus=4,
        chunks=2,
        print_stderr=True,
        print_stdout=False,
        usecache=True,
        with_sendevent=False,
    )
    print(df)
    ```

    Note:
    - This function uses the C-extension https://github.com/hansalemaos/chopchopcolorc for efficient parallel color search.
    - The resulting DataFrame includes an 'aa_colorsearch_c' column containing search results.
    - If a color is found in an image, the corresponding entry is a non-negative integer; otherwise, it is NA.
    - Additional columns include 'aa_csearch_mean_x' and 'aa_csearch_mean_y', representing the mean x and y coordinates of the found colors.
    - If 'with_sendevent' is True, the DataFrame includes columns for input tap and sendevent information.
    """
    with_input_tap = True
    df2 = df.copy()
    df2.loc[:, "aa_colorsearch_c"] = pd.Series(
        color_search_c(
            pics=df2[column].to_list(),
            rgb_tuples=colors2find,
            cpus=cpus,
            chunks=chunks,
            print_stderr=print_stderr,
            print_stdout=print_stdout,
            usecache=usecache,
        )
    ).ds_apply_ignore(
        pd.NA, lambda x: pd.NA if len(x) == 1 and np.sum(x[0]) <= 0 else x
    )

    df2["aa_csearch_mean_x"] = df2.aa_colorsearch_c.ds_apply_ignore(
        pd.NA,
        lambda q: numexpr.evaluate(
            "h/j",
            global_dict={},
            local_dict={
                "j": len(q),
                "h": numexpr.evaluate(
                    "sum(q)", local_dict={"q": q[..., 1]}, global_dict={}
                ),
            },
        ),
    ).fillna(-1)

    df2["aa_csearch_mean_y"] = df2.aa_colorsearch_c.ds_apply_ignore(
        pd.NA,
        lambda q: numexpr.evaluate(
            "h/j",
            global_dict={},
            local_dict={
                "j": len(q),
                "h": numexpr.evaluate(
                    "sum(q)", local_dict={"q": q[..., 0]}, global_dict={}
                ),
            },
        ),
    ).fillna(-1)
    #
    df2["aa_csearch_mean_y"] = df2.ds_apply_ignore(
        pd.NA, lambda j: int(j.aa_csearch_mean_y + j.aa_start_y), axis=1
    )
    df2["aa_csearch_mean_x"] = df2.ds_apply_ignore(
        pd.NA, lambda j: int(j.aa_csearch_mean_x + j.aa_start_x), axis=1
    )
    dfr = DataFrameWithMeta(
        df2,
        adb_instance=df.adb_instance,
    )
    dfrclick = _add_click_methods(
        dfr,
        with_input_tap=with_input_tap,
        with_sendevent=with_sendevent,
        column_input_tap="ff_colormean_input_tap",
        column_x="aa_csearch_mean_x",
        column_y="aa_csearch_mean_y",
        sendevent_prefix="ff_colormean_sendevent_",
    )
    dfrclick.loc[
        ((dfrclick["aa_colorsearch_c"].isna())),
        [
            "aa_csearch_mean_x",
            "aa_csearch_mean_y",
            "ff_colormean_input_tap",
            *[
                k
                for k in dfrclick.columns
                if str(k).startswith("ff_colormean_sendevent_")
            ],
        ],
    ] = pd.NA

    return dfrclick


def _pandas_ex_find_shapes(
    dframe,
    with_draw_function=True,
    threshold1=10,
    threshold2=90,
    approxPolyDPvar=0.01,
    cpus=5,
    chunks=1,
    print_stderr=True,
    print_stdout=False,
    usecache=True,
    column="aa_screenshot",
    with_sendevent=False,
):
    r"""
    Find and analyze shapes in images stored in a DataFrame column.

    Parameters:
    - dframe (DataFrameWithMeta): The input DataFrame containing the data.
    - with_draw_function (bool, optional): Whether to include a drawing function in the results. Defaults to True.
    - threshold1 (int, optional): The first threshold value for shape detection. Defaults to 10.
    - threshold2 (int, optional): The second threshold value for shape detection. Defaults to 90.
    - approxPolyDPvar (float, optional): The approximation accuracy for polygonal shapes. Defaults to 0.01.
    - cpus (int, optional): The number of CPU cores to use for parallel processing. Defaults to 5.
    - chunks (int, optional): The number of chunks to divide the data into for parallel processing. Defaults to 1.
    - print_stderr (bool, optional): Whether to print stderr messages during shape analysis. Defaults to True.
    - print_stdout (bool, optional): Whether to print stdout messages during shape analysis. Defaults to False.
    - usecache (bool, optional): Whether to use caching for shape analysis results. Defaults to True.
    - column (str, optional): The column in the DataFrame containing images.
      Defaults to 'aa_screenshot' if not specified.
    - with_sendevent (bool, optional): Whether to include sendevent information in the result. Defaults to False.

    Returns:
    - DataFrameWithMeta: The input DataFrame with additional columns related to shape analysis.

    Example:
    df = df.plus_find_shapes(
        with_draw_function=True,
        threshold1=20,
        threshold2=80,
        approxPolyDPvar=0.005,
        cpus=4,
        chunks=2,
        print_stderr=True,
        print_stdout=False,
        usecache=True,
        column='aa_screenshot',
        with_sendevent=False,
    )
    print(df)

    Note:
    - This function uses https://github.com/hansalemaos/multiprocshapefinder
      to detect and analyze shapes in images.
    - The resulting DataFrame includes columns related to shape coordinates and characteristics.
    - If 'with_draw_function' is True, a 'ff_drawn_shape' column includes drawn shapes using the
     DrawExecutor class.
    - Additional columns include 'aa_offset_x' and 'aa_offset_y', representing the offset of the detected shapes.
    - 'aa_real_center_x' and 'aa_real_center_y' represent the real center coordinates of the detected shapes.
    - If 'with_sendevent' is True, the DataFrame includes columns for input tap and sendevent information.
    """
    with_input_tap = True
    lookupdict = dframe.index.to_frame().reset_index(drop=True).to_dict()[0]
    df = dframe.reset_index(drop=True)
    imagefilter = df[column].to_dict()  #

    df4 = find_all_shapes(
        list(imagefilter.values()),
        threshold1=threshold1,
        threshold2=threshold2,
        approxPolyDPvar=approxPolyDPvar,
        cpus=cpus,
        chunks=chunks,
        print_stderr=print_stderr,
        print_stdout=print_stdout,
        usecache=usecache,
    )
    df4 = df4.astype(
        {
            k: "Int64"
            for k in df4.columns
            if str(k).startswith("aa_bound") or str(k).startswith("aa_center")
        }
    )

    if with_draw_function:
        allclasses = []
        for i1, i2 in zip(range(len(df4)), range(1, len(df4) + 1)):
            df4t = df4.iloc[i1:i2]
            allclasses.append(
                DrawExecutor(
                    draw_results, df4t, df.iloc[df4t.aa_img_index.iloc[0]][column]
                )
            )
        df4["ff_drawn_shape"] = allclasses
    df4.loc[:, "aa_img_index"] = df4["aa_img_index"].map(lookupdict)

    df4["aa_offset_x"] = (
        df4.groupby("aa_img_index")
        .apply(lambda x: dframe.loc[x.aa_img_index])
        .aa_start_x.reset_index()["aa_start_x"]
    )
    df4["aa_offset_y"] = (
        df4.groupby("aa_img_index")
        .apply(lambda x: dframe.loc[x.aa_img_index])
        .aa_start_y.reset_index()["aa_start_y"]
    )

    df4["aa_real_center_x"] = df4["aa_center_x"] + df4["aa_offset_x"]
    df4["aa_real_center_y"] = df4["aa_center_y"] + df4["aa_offset_y"]

    dfr = DataFrameWithMeta(
        df4,
        adb_instance=dframe.adb_instance,
    )
    return _add_click_methods(
        dfr,
        with_input_tap=with_input_tap,
        with_sendevent=with_sendevent,
        column_input_tap="ff_shape_input_tap",
        column_x="aa_real_center_x",
        column_y="aa_real_center_y",
        sendevent_prefix="ff_shape_sendevent_",
    )


def _pandas_ex_template_matching(
    df,
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
):
    r"""
    Perform template matching on images in a DataFrame column.

    Parameters:
    - df (DataFrameWithMeta): The input DataFrame containing the data.
    - needles (list): A dict (name : image) of template images to match.
    - with_sendevent (bool, optional): Whether to include sendevent commands in the results. Defaults to True.
    - with_image_data (bool, optional): Whether to include image data in the results. Defaults to True.
    - thresh (float, optional): The threshold for matching similarity. Defaults to 0.9.
    - pad_input (bool, optional): Whether to pad the input images. Defaults to False.
    - mode (str, optional): The padding mode if 'pad_input' is True. Defaults to 'constant'.
    - constant_values (int, optional): The constant value for padding. Defaults to 0.
    - usecache (bool, optional): Whether to use caching for template matching results. Defaults to True.
    - processes (int, optional): The number of CPU cores to use for parallel processing. Defaults to 5.
    - chunks (int, optional): The number of chunks to divide the data into for parallel processing. Defaults to 1.
    - print_stdout (bool, optional): Whether to print stdout messages during template matching. Defaults to False.
    - print_stderr (bool, optional): Whether to print stderr messages during template matching. Defaults to True.
    - column (str, optional): The column in the DataFrame containing images.
      Defaults to 'aa_screenshot' if not specified.

    Returns:
    - DataFrameWithMeta: A DataFrame with matched template information, including input taps and sendevent commands.

    Example:
    ```
    df = df.plus_template_matching(
        needles={'b1': 'c:\pic1.png', 'b2': 'c:\pic2.png'},
        with_sendevent=True,
        with_image_data=True,
        thresh=0.8,
        pad_input=True,
        mode='constant',
        constant_values=255,
        usecache=True,
        processes=4,
        chunks=2,
        print_stdout=True,
        print_stderr=False,
        column='aa_screenshot',
    )
    print(df)
    ```

    Note:
    - This function uses the https://github.com/hansalemaos/needlefinder algorithm for template matching.
    - The resulting DataFrame includes columns related to template matching results, including input taps and sendevent commands.
    - The 'needles' parameter should be a list of template images.
    - Input taps and sendevent commands are included if 'with_input_tap' and 'with_sendevent' are True.
    """
    with_input_tap = True
    self = df.adb_instance
    df2 = df.dropna(subset=column).copy()
    df2["aa_realindex"] = df2.index.__array__().copy()
    haystacks = df2[column].to_list()
    dfneedles = find_needles_in_multi_haystacks(
        haystacks=haystacks,
        needles=needles,
        with_image_data=with_image_data,
        thresh=thresh,
        pad_input=pad_input,
        mode=mode,
        constant_values=constant_values,
        usecache=usecache,
        processes=processes,
        chunks=chunks,
        print_stdout=print_stdout,
        print_stderr=print_stderr,
    )
    lookupdict = df2.aa_realindex.to_frame().reset_index(drop=True).to_dict()
    dfneedlesabs = dfneedles.groupby("aa_img_index", group_keys=False).apply(
        lambda q: DataFrameWithMeta(
            [
                q.aa_start_x + df2.iloc[q.aa_img_index].aa_start_x.iloc[0],
                q.aa_start_y + df2.iloc[q.aa_img_index].aa_start_y.iloc[0],
                q.aa_end_x + df2.iloc[q.aa_img_index].aa_start_x.iloc[0],
                q.aa_end_y + df2.iloc[q.aa_img_index].aa_start_y.iloc[0],
                q.aa_center_x + df2.iloc[q.aa_img_index].aa_start_x.iloc[0],
                q.aa_center_y + df2.iloc[q.aa_img_index].aa_start_y.iloc[0],
                q.aa_img_index,
            ],
            adb_instance=self,
        ).T
    )
    dfneedlesabs.attrs["adb_instance"] = self

    for col in dfneedlesabs.columns:
        try:
            dfneedlesabs[col] = dfneedlesabs[col].astype("Int64")
        except Exception:
            pass
    dfneedlesabs["aa_img_index"] = dfneedlesabs["aa_img_index"].map(
        lookupdict["aa_realindex"]
    )

    dfneedlesabs.columns = [f"aa_needle_abs_{x[3:]}" for x in dfneedlesabs.columns]
    dfneedles.columns = [f"aa_needle_{x[3:]}" for x in dfneedles.columns]
    dfneedlesabs2 = pd.concat([dfneedlesabs, dfneedles], axis=1)
    df2["aa_merge_index"] = df2.index.__array__().copy()
    dfr = (
        df2.merge(
            dfneedlesabs2,
            right_on="aa_needle_abs_img_index",
            left_on="aa_merge_index",
            how="inner",
        )
        .drop(columns="aa_merge_index")
        .rename(columns={"aa_needle_abs_img_index": "aa_img_index"})
        .reset_index(drop=True)
    )
    dfr = DataFrameWithMeta(
        dfr,
        adb_instance=self,
    )
    return _add_click_methods(
        dfr,
        with_input_tap=with_input_tap,
        with_sendevent=with_sendevent,
        column_input_tap="ff_needle_input_tap",
        column_x="aa_needle_center_x",
        column_y="aa_needle_center_y",
        sendevent_prefix="ff_needle_sendevent_",
    )


def _get_pandas_ex_color_cluster(
    img,
    colors,
    reverse_colors=True,
    backend="C",
    memorylimit_mb=10000,
    eps=3,
    min_samples=10,
    algorithm="auto",
    leaf_size=30,
    n_jobs=5,
    max_width=100,
    max_height=100,
    interpolation=cv2.INTER_NEAREST,
):
    r"""
    Perform color clustering on an image and return the cluster information.

    Parameters:
    - img (numpy.ndarray): The input image for color clustering.
    - colors (list): List of colors to identify clusters.
    - reverse_colors (bool, optional): Whether to reverse the colors (RGB - BGR). Defaults to True.
    - backend (str, optional): The backend for Euclidean distance calculation. Defaults to "C".
    - memorylimit_mb (int, optional): Memory limit in megabytes for the Euclidean distance matrix.
      Defaults to 10000.
    - eps (float, optional): The maximum distance between two samples for one to be considered
      as in the neighborhood of the other.
      Defaults to 3.
    - min_samples (int, optional): The number of samples in a neighborhood
      for a point to be considered as a core point. Defaults to 10.
    - algorithm (str, optional): The algorithm to compute nearest neighbors. Defaults to "auto".
    - leaf_size (int, optional): Leaf size passed to BallTree or KDTree. Defaults to 30.
    - n_jobs (int, optional): The number of parallel jobs to run for DBSCAN. Defaults to 5.
    - max_width (int, optional): Maximum width for resizing the image. Defaults to 100.
    - max_height (int, optional): Maximum height for resizing the image. Defaults to 100.
    - interpolation (int, optional): Interpolation method for resizing the image.
      Defaults to cv2.INTER_NEAREST.

    Returns:
    - ShapelyGeometryCollection: A Shapely Geometry Collection representing the color clusters.


    Note:
    - This function utilizes the ColorCluster class for color clustering and DBSCAN for clustering.
    - The resulting Shapely Geometry Collection represents the identified color clusters in the image.
    ```
    """
    return (
        ColorCluster(
            img=img,
            max_width=max_width,
            max_height=max_height,
            interpolation=interpolation,
        )
        .find_colors(colors=colors, reverse_colors=reverse_colors)
        .calculate_euclidean_matrix(backend=backend, memorylimit_mb=memorylimit_mb)
        .get_dbscan_labels(
            eps=eps,
            min_samples=min_samples,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
        )
        .get_clusters()
        .get_shapely()
    )


def _pandas_ex_color_cluster(
    df,
    colors,
    reverse_colors=True,
    backend="C",
    memorylimit_mb=10000,
    eps=10,
    min_samples=3,
    algorithm="auto",
    leaf_size=30,
    n_jobs=5,
    max_width=100,
    max_height=100,
    interpolation=cv2.INTER_NEAREST,
    column="aa_screenshot",
    with_sendevent=False,
):
    r"""
    Perform color clustering on images in a DataFrame column and add the cluster information to the DataFrame.

    Parameters:
    - df (DataFrameWithMeta): Input DataFrame with image data and metadata.
    - colors (list): List of colors to identify clusters.
    - reverse_colors (bool, optional): Whether to reverse the order of colors. Defaults to True.
    - backend (str, optional): The backend for Euclidean distance calculation. Defaults to "C".
    - memorylimit_mb (int, optional): Memory limit in megabytes for the Euclidean distance matrix.
      Defaults to 10000.
    - eps (float, optional): The maximum distance between two samples for one to be considered as in
      the neighborhood of the other. Defaults to 10.
    - min_samples (int, optional): The number of samples in a neighborhood for a point to be
      considered as a core point. Defaults to 3.
    - algorithm (str, optional): The algorithm to compute nearest neighbors. Defaults to "auto".
    - leaf_size (int, optional): Leaf size passed to BallTree or KDTree. Defaults to 30.
    - n_jobs (int, optional): The number of parallel jobs to run for DBSCAN. Defaults to 5.
    - max_width (int, optional): Maximum width for resizing the image. Defaults to 100.
    - max_height (int, optional): Maximum height for resizing the image. Defaults to 100.
    - interpolation (int, optional): Interpolation method for resizing the image.
      Defaults to cv2.INTER_NEAREST.
    - column (str, optional): The column containing image data. Defaults to "aa_screenshot".
    - with_sendevent (bool, optional): Whether to include sendevent methods in the output DataFrame.
      Defaults to False.

    Returns:
    - DataFrameWithMetaShaply: DataFrame with added color cluster information and methods for interaction
     with the clusters.

    Example:
    ```
    df = DataFrameWithMeta()
    df["aa_screenshot"] = [...]  # Add image data to the DataFrame
    df["aa_start_x"] = [...]  # Add metadata to the DataFrame

    clustered_df = _pandas_ex_color_cluster(
        df=df,
        colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        reverse_colors=True,
        backend="C",
        memorylimit_mb=10000,
        eps=10,
        min_samples=3,
        algorithm="auto",
        leaf_size=30,
        n_jobs=5,
        max_width=100,
        max_height=100,
        interpolation=cv2.INTER_NEAREST,
        column="aa_screenshot",
        with_sendevent=True,
    )

    print(clustered_df)
    ```

    Note:
    - This function applies color clustering to images in the specified DataFrame column.
    - The resulting DataFrame includes additional columns with color cluster information and methods for interaction.
    ```
    """
    with_input_tap = True
    aa_color_cluster = df[column].ds_apply_ignore(
        pd.NA,
        lambda img: _get_pandas_ex_color_cluster(
            img,
            colors,
            reverse_colors=reverse_colors,
            backend=backend,
            memorylimit_mb=memorylimit_mb,
            eps=eps,
            min_samples=min_samples,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
            max_width=max_width,
            max_height=max_height,
            interpolation=interpolation,
        ),
    )
    importantdata = []
    for i, item in zip(df.index, aa_color_cluster):
        if not pd.isna(item):
            importantdata.append(
                pd.DataFrame(item.shapelydata).T.assign(
                    aa_img_index=i, aa_colorcluster=item
                )
            )
    dfshapely = pd.concat(importantdata, ignore_index=True)

    dfshapely["aa_center_cluster_x"] = dfshapely.ds_apply_ignore(
        pd.NA,
        lambda x: x["representative_point"][0] + df.loc[x.aa_img_index].aa_start_x,
        axis=1,
    )
    dfshapely["aa_center_cluster_y"] = dfshapely.ds_apply_ignore(
        pd.NA,
        lambda x: x["representative_point"][1] + df.loc[x.aa_img_index].aa_start_y,
        axis=1,
    )

    dfshapely = DataFrameWithMeta(
        dfshapely,
        adb_instance=df.adb_instance,
    )
    shapelydf = _add_click_methods(
        dfshapely,
        with_input_tap=with_input_tap,
        with_sendevent=with_sendevent,
        column_input_tap="ff_cluster_input_tap",
        column_x="aa_center_cluster_x",
        column_y="aa_center_cluster_y",
        sendevent_prefix="ff_cluster_sendevent_",
    )
    dfz = df.merge(shapelydf, left_index=True, right_on="aa_img_index").reset_index(
        drop=True
    )
    dfx = DataFrameWithMetaShaply(dfz, adb_instance=dfz.adb_instance)
    dfx.attrs["adb_instance"] = dfz.adb_instance
    return dfx


def _add_click_methods(
    df,
    with_input_tap=True,
    with_sendevent=True,
    column_input_tap="ff_input_tap",
    column_x="aa_center_x",
    column_y="aa_center_y",
    sendevent_prefix="ff_sendevent_",
):
    r"""
    Add click methods to a DataFrame for interaction with specified coordinates.

    Parameters:
    - df (DataFrameWithMeta): Input DataFrame with data and metadata.
    - with_input_tap (bool, optional): Whether to include input tap methods. Defaults to True.
    - with_sendevent (bool, optional): Whether to include sendevent methods. Defaults to True.
    - column_input_tap (str, optional): The column name for input tap methods. Defaults to "ff_input_tap".
    - column_x (str, optional): The column containing x-coordinate values. Defaults to "aa_center_x".
    - column_y (str, optional): The column containing y-coordinate values. Defaults to "aa_center_y".
    - sendevent_prefix (str, optional): The prefix for sendevent method columns. Defaults to "ff_sendevent_".

    Returns:
    - DataFrameWithMeta: DataFrame with added click methods.

    Example:
    ```
    df = DataFrameWithMeta()
    df["aa_center_x"] = [...]  # Add x-coordinate data to the DataFrame
    df["aa_center_y"] = [...]  # Add y-coordinate data to the DataFrame

    df_with_click_methods = _add_click_methods(
        df=df,
        with_input_tap=True,
        with_sendevent=True,
        column_input_tap="ff_input_tap",
        column_x="aa_center_x",
        column_y="aa_center_y",
        sendevent_prefix="ff_sendevent_",
    )

    print(df_with_click_methods)
    ```

    Note:
    - This function adds click methods to a DataFrame based on specified columns containing x and y coordinates.
    - Click methods include input tap and sendevent methods for interacting with the specified coordinates.
    - The resulting DataFrame includes additional columns with click methods.
    """
    self = df.adb_instance
    if with_input_tap:
        df[column_input_tap] = df.ds_apply_ignore(
            pd.NA,
            lambda q: Clicker(self.sh_input_tap, q[column_x], q[column_y])
            if not pd.isna(q[column_x]) and not pd.isna(q[column_y])
            else pd.NA,
            axis=1,
        )

    if with_sendevent:
        if not self._sendevent_devices:
            self._sendevent_devices = self.sh_get_sendevent_input_devices()
        self._get_height_width()
        for de in self._sendevent_devices:
            df[f'{sendevent_prefix}{de[0].split("/")[-1]}'] = df.ds_apply_ignore(
                pd.NA,
                lambda q: Clicker(
                    self.sh_sendevent_touch,
                    x=q[column_x],
                    y=q[column_y],
                    inputdev=de[0].split("/")[-1],
                    inputdevmax=int(de[1]),
                    width=self.device_width,
                    height=self.device_height,
                )
                if not pd.isna(q[column_x]) and not pd.isna(q[column_y])
                else pd.NA,
                axis=1,
            )
    return df


def _tesseract(
    df,
    column="aa_screenshot",
    tesser_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    add_after_tesseract_path="",
    add_at_the_end="-l eng+por --psm 3",
    processes=5,
    chunks=1,
    print_stdout=False,
    print_stderr=False,
    with_sendevent=False,
):
    r"""
    Perform OCR (Optical Character Recognition) using Tesseract on the specified image column of a DataFrame.

    Parameters:
    - df (DataFrameWithMeta): Input DataFrame with data and metadata.
    - column (str, optional): The column containing images for OCR. Defaults to "aa_screenshot".
    - tesser_path (str, optional): Path to the Tesseract executable. Defaults to
      "C:\Program Files\Tesseract-OCR\tesseract.exe".
    - add_after_tesseract_path (str, optional): Additional Tesseract executable arguments to be added
      after the path. Defaults to an empty string.
    - add_at_the_end (str, optional): Additional Tesseract executable arguments to be added at the end.
      Defaults to "-l eng+por --psm 3".
    - processes (int, optional): Number of processes to use for parallel OCR. Defaults to 5.
    - chunks (int, optional): Number of chunks to divide the OCR process into. Defaults to 1.
    - print_stdout (bool, optional): Whether to print standard output during OCR. Defaults to False.
    - print_stderr (bool, optional): Whether to print standard error during OCR. Defaults to False.
    - with_sendevent (bool, optional): Whether to include sendevent methods in the output DataFrame.
      Defaults to False.

    Returns:
    - DataFrameWithMeta: DataFrame with added OCR results and metadata.

    Example:
    df_with_ocr = _tesseract(
        df=df,
        column="aa_screenshot",
        tesser_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        add_after_tesseract_path="",
        add_at_the_end="-l eng+por --psm 3",
        processes=5,
        chunks=1,
        print_stdout=False,
        print_stderr=False,
        with_sendevent=False,
    )

    print(df_with_ocr)

    Note:
    - This function performs OCR on images in the specified column using Tesseract.
    - Additional Tesseract arguments can be provided to customize the OCR process.
    - The resulting DataFrame includes additional columns with OCR results and metadata.
    """
    nona = df[column].dropna().to_dict()
    dftes = tesser_ocr(
        piclist=list(nona.values()),
        tesser_path=tesser_path,
        add_after_tesseract_path=add_after_tesseract_path,
        add_at_the_end=add_at_the_end,
        processes=processes,
        chunks=chunks,
        print_stdout=print_stdout,
        print_stderr=print_stderr,
    )
    lookupdict = {k1: k for k1, k in enumerate(nona.keys())}
    dftes.aa_document_index = dftes.aa_document_index.map(lookupdict)
    dftes.columns = [f"tt_{x[3:]}" for x in dftes.columns]
    dfttt = df.merge(dftes, right_on="tt_document_index", left_index=True)
    dfttt.reset_index(drop=True, inplace=True)
    dfttt["tt_center2click_x"] = dfttt.apply(
        lambda x: x.aa_start_x + x.tt_center_x, axis=1
    )
    dfttt["tt_center2click_y"] = dfttt.apply(
        lambda x: x.aa_start_y + x.tt_center_y, axis=1
    )
    dftttn = DataFrameWithMeta(
        dfttt,
        adb_instance=df.adb_instance,
    )
    with_input_tap = True

    return _add_click_methods(
        dftttn,
        with_input_tap=with_input_tap,
        with_sendevent=with_sendevent,
        column_input_tap="tt_input_tap",
        column_x="tt_center2click_x",
        column_y="tt_center2click_y",
        sendevent_prefix="tt_sendevent_",
    )


def activate_pandas_extensions(
    modules=(
        "plus_find_shapes",
        "plus_template_matching",
        "plus_color_search_c",
        "plus_color_cluster",
        "plus_count_all_colors",
        "plus_count_all_colors_coords",
        "plus_fuzzy_merge",
        "plus_tesser_act",
    )
):
    r"""
    Activate custom extensions for the DataFrameWithMeta class based on specified modules.
    plus_find_shapes - pip install multiprocshapefinder
    plus_template_matching - pip install needlefinder
    plus_color_search_c - pip install chopchopcolorc
    plus_color_cluster - pip install locatecolorcluster
    plus_count_all_colors - pip install colorcountcython
    plus_count_all_colors_coords - pip install colorcountcython
    plus_fuzzy_merge - pip install a_pandas_ex_fuzzymerge
    plus_tesser_act - pip install multitessiocr

    Parameters:
    - modules (tuple, optional): A tuple of strings specifying the modules to activate. Defaults to a tuple
      containing various extension modules such as "plus_save_screenshots" and others.

    Returns:
    None

    Example:
    ```
    activate_pandas_extensions(
        modules=("plus_save_screenshots", "plus_template_matching", "plus_color_search_c")
    )
    ```

    Note:
    - This function activates custom extensions for the DataFrameWithMeta class based on specified modules.
    - Each module corresponds to a specific extension and must be installed separately using pip.
    - If a module is not installed, an error message will be displayed in the standard error stream.
    """
    try:
        if "plus_find_shapes" in modules:
            exec(
                """from multiprocshapefinder import find_all_shapes, draw_results""",
                globals(),
            )
        DataFrameWithMeta.plus_find_shapes = _pandas_ex_find_shapes
    except Exception as e:
        sys.stderr.write(
            'Module could not be loaded! Use "pip install multiprocshapefinder" to install it'
        )
        sys.stderr.flush()
    try:
        if "plus_template_matching" in modules:
            exec(
                """from needlefinder import find_needles_in_multi_haystacks""",
                globals(),
            )
            DataFrameWithMeta.plus_template_matching = _pandas_ex_template_matching
    except Exception as e:
        sys.stderr.write(
            'Module could not be loaded! Use "pip install needlefinder" to install it'
        )
        sys.stderr.flush()
    try:
        if "plus_color_search_c" in modules:
            exec(
                """from chopchopcolorc import color_search_c""",
                globals(),
            )
            DataFrameWithMeta.plus_color_search_c = _pandas_ex_color_search_with_c
    except Exception as e:
        sys.stderr.write(
            'Module could not be loaded! Use "pip install chopchopcolorc" to install it'
        )
        sys.stderr.flush()
    try:
        if "plus_color_cluster" in modules:
            exec(
                """from locatecolorcluster import ColorCluster""",
                globals(),
            )
            DataFrameWithMeta.plus_color_cluster = _pandas_ex_color_cluster
    except Exception as e:
        sys.stderr.write(
            'Module could not be loaded! Use "pip install locatecolorcluster" to install it'
        )
        sys.stderr.flush()
    try:
        if "plus_count_all_colors" in modules:
            exec(
                """from colorcountcython import colorcount""",
                globals(),
            )
            DataFrameWithMeta.plus_count_all_colors = _pandas_ex_count_colors
    except Exception as e:
        sys.stderr.write(
            'Module could not be loaded! Use "pip install colorcountcython" to install it'
        )
        sys.stderr.flush()
    try:
        if "plus_count_all_colors_coords" in modules:
            exec(
                """from colorcountcython import colorcount""",
                globals(),
            )
            DataFrameWithMeta.plus_count_all_colors_coords = _pandas_ex_color_coords
    except Exception as e:
        sys.stderr.write(
            'Module could not be loaded! Use "pip install colorcountcython" to install it'
        )
        sys.stderr.flush()
    try:
        if "plus_fuzzy_merge" in modules:
            exec(
                """from a_pandas_ex_fuzzymerge import pd_add_fuzzymerge""",
                globals(),
            )
            exec(
                """from rapidfuzz import fuzz""",
                globals(),
            )
            pd_add_fuzzymerge()

            DataFrameWithMeta.plus_fuzzy_merge = _pandas_ex_fuzzy_match
    except Exception as e:
        sys.stderr.write(
            'Module could not be loaded! Use "pip install a_pandas_ex_fuzzymerge" to install it'
        )
        sys.stderr.flush()
    try:
        if "plus_tesser_act" in modules:
            exec(
                """from multitessiocr import tesser_ocr""",
                globals(),
            )

            DataFrameWithMeta.plus_tesser_act = _tesseract
    except Exception as e:
        sys.stderr.write(
            'Module could not be loaded! Use "pip install multitessiocr" to install it'
        )
        sys.stderr.flush()


def cropimage(img, coords):
    try:
        return img[coords[1] : coords[3], coords[0] : coords[2]]
    except Exception as e:
        sys.stderr.write(f"{e}")
        sys.stderr.flush()
        return pd.NA


class Clicker:
    r"""
    Represents a click action on specific coordinates.

    Parameters:
    - fu (callable): The function to perform the click action.
    - x (int): The x-coordinate of the click.
    - y (int): The y-coordinate of the click.
    - **kwargs: Additional keyword arguments to be passed to the click function.

    Methods:
    - __call__(offset_x=0, offset_y=0, **kwargs): Perform the click action with optional
      offset values and additional keyword arguments.
    - __str__(): Return a string representation of the Clicker object.
    - __repr__(): Return a string representation of the Clicker object.

    """

    def __init__(self, fu, x, y, **kwargs):
        self.fu = fu
        self.x = x
        self.y = y
        self.kwargs = kwargs

    def __call__(self, offset_x=0, offset_y=0, **kwargs):
        r"""
        Perform the click action with optional offset values and additional keyword arguments.

        Parameters:
        - offset_x (int, optional): The x-coordinate offset for the click action. Defaults to 0.
        - offset_y (int, optional): The y-coordinate offset for the click action. Defaults to 0.
        - **kwargs: Additional keyword arguments to be passed to the click function.

        Returns:
        The result of the click action.
        """
        kwold = self.kwargs.copy()
        kwold.update(kwargs)
        return self.fu(self.x + offset_x, self.y + offset_y, **kwold)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __repr__(self):
        return self.__str__()


class AdbControlPlus(AdbControl):
    def __init__(
        self,
        adb_path,
        device_serial,
        use_busybox=False,
        connect_to_device=True,
        invisible=True,
        print_stdout=True,
        print_stderr=True,
        limit_stdout=3,
        limit_stderr=3,  # limits the history of shellcommands - can be checked at blocking_shell.stderr
        limit_stdin=None,
        convert_to_83=True,
        wait_to_complete=0.1,
        flush_stdout_before=False,
        flush_stdin_before=False,
        flush_stderr_before=False,
        exitcommand="xxxCOMMANDxxxDONExxx",
        capture_stdout_stderr_first=True,
        global_cmd=True,
        global_cmd_timeout=10,
        use_eval=True,
        eval_timeout=10,
        device_height=0,
        device_width=0,
    ):
        r"""
        Extended AdbControl class with additional functionalities.

        Parameters:
        - adb_path (str): Path to the ADB executable.
        - device_serial (str): Serial number of the Android device.
        - use_busybox (bool): Use BusyBox if available. Defaults to False.
        - connect_to_device (bool): Connect to the device. Defaults to True.
        - invisible (bool): Run a hidden shell (important when compiling with Nuitka). Defaults to True.
        - print_stdout (bool): Print stdout of ADB commands. Defaults to True.
        - print_stderr (bool): Print stderr of ADB commands. Defaults to True.
        - limit_stdout (int): Limit the history of stdout in shell commands. Defaults to 3.
        - limit_stderr (int): Limit the history of stderr in shell commands. Defaults to 3.
        - limit_stdin (int): Limit the history of stdin in shell commands.
        - convert_to_83 (bool): Convert to 8.3 (Windows). Defaults to True.
        - wait_to_complete (int): Sleep for a specified duration before checking output. Defaults to 0.1.
        - flush_stdout_before (bool): Flush stdout before executing a command. Defaults to False.
        - flush_stdin_before (bool): Flush stdin before executing a command. Defaults to False.
        - flush_stderr_before (bool): Flush stderr before executing a command. Defaults to False.
        - exitcommand (str): Exit command for ADB. Defaults to "xxxCOMMANDxxxDONExxx" - must be a string that never shows up in any ADB result.
        - capture_stdout_stderr_first (bool): Capture stdout and stderr before executing a command. Defaults to True.
        - global_cmd (bool): Use global commands. Defaults to True.
        - global_cmd_timeout (int): Timeout for global commands. Defaults to 10.
        - use_eval (bool): Use eval for shell commands. Defaults to True.
        - eval_timeout (int): Timeout for eval commands. Defaults to 10.
        - device_height (int): Height of the Android device screen. Defaults to 0.
        - device_width (int): Width of the Android device screen. Defaults to 0.

        Methods:
        - plus_record_events(print_output=False, print_output_pandas=False, convert_to_pandas=True):
            Start recording events and return an EventRecord object.

        - plus_record_and_save_to_sdcard(tmpfolder_device, tmpfolder_local=None, device='event4', print_output=True,
            add_closing_command=True, clusterevents=16):
            Record events and save them to the SD card.

        - plus_start_logcat(logfile=None, devices=None):
            Start the logcat process and return the subprocess.Popen object.

        - plus_start_fast_screenshot_iter(bitrate="20M", screenshotbuffer=10, go_idle=0):
            Start iterating through fast screenshots and yield each image.

        - plus_uidump(timeout=60, with_screenshot=True, screenshot=None, nice=False, su=False,
            with_input_tap=True, with_sendevent=True):
            Capture UI dump and return the result as a DataFrameWithMeta.

        - plus_uidump_with_freeze(procregex_for_lsof, timeout=60, with_screenshot=True, screenshot=None,
            with_input_tap=True, with_sendevent=True):
            Capture UI dump with process freezing and return the result as a DataFrameWithMeta.

        - plus_screenshot_as_np():
            Capture a screenshot and return it as a NumPy array.

        - plus_activity_elements_dump(with_screenshot=True, screenshot=None, with_input_tap=True, with_sendevent=True):
            Dump activity elements and return the result as a DataFrameWithMeta.

        Attributes:
        - device_height (int): Height of the Android device screen.
        - device_width (int): Width of the Android device screen.


        """
        super().__init__(
            adb_path=adb_path,
            device_serial=device_serial,
            use_busybox=use_busybox,
            connect_to_device=connect_to_device,
            invisible=invisible,
            print_stdout=print_stdout,
            print_stderr=print_stderr,
            limit_stdout=limit_stdout,
            limit_stderr=limit_stderr,
            limit_stdin=limit_stdin,
            convert_to_83=convert_to_83,
            wait_to_complete=wait_to_complete,
            flush_stdout_before=flush_stdout_before,
            flush_stdin_before=flush_stdin_before,
            flush_stderr_before=flush_stderr_before,
            exitcommand=exitcommand,
            capture_stdout_stderr_first=capture_stdout_stderr_first,
            global_cmd=global_cmd,
            global_cmd_timeout=global_cmd_timeout,
            use_eval=use_eval,
            eval_timeout=eval_timeout,
        )
        self.device_height = device_height
        self.device_width = device_width
        self._sendevent_devices = []
        self.hashvalue = hash((time.time(), device_serial))
        adbconfig.all_devices[self.hashvalue] = self

    def __hash__(self):
        return self.hashvalue

    def plus_record_events(
        self,
        print_output=False,
        print_output_pandas=False,
        convert_to_pandas=True,
    ):
        r"""
        Start recording events using EventRecord.

        Parameters:
        - print_output (bool): Print the output of ADB commands. Defaults to False.
        - print_output_pandas (bool): Print the pandas DataFrame of recorded events. Defaults to False.
        - convert_to_pandas (bool): Convert the output to pandas DataFrame. Defaults to True.

        Returns:
        EventRecord: An instance of the EventRecord class for recording events.

        Example:
        adb_plus = AdbControlPlus(adb_path="/path/to/adb", device_serial="123456")
        recorder = adb_plus.plus_record_events(print_output=True, print_output_pandas=True)
        recorder.start_recording()
        for x in recorder.resultsdf:
            print(x)
        recorder.stop = True

        """
        return EventRecord(
            adb_path=self.adbpath,
            device_serial=self.device_serial,
            print_output=print_output,
            print_output_pandas=print_output_pandas,
            convert_to_pandas=convert_to_pandas,
            parent1replacement="\x80",
            parent2replacement="\x81",
        )

    def plus_record_and_save_to_sdcard(
        self,
        tmpfolder_device,
        tmpfolder_local=None,
        device="event4",
        print_output=True,
        add_closing_command=True,
        clusterevents=16,
    ):
        r"""
        Record Android events and save to the specified folders.

        Parameters:
        - tmpfolder_device (str): The temporary folder on the device to store recorded events.
        - tmpfolder_local (str): The local folder on the computer to save recorded events. Defaults to None.
        - device (str): The device identifier. Defaults to "event4".
        - print_output (bool): Print the output of ADB commands. Defaults to True.
        - add_closing_command (bool): Add a closing command to stop event recording. Defaults to True.
        - clusterevents (int): Number of events to cluster before saving. Defaults to 16.

        Returns:
        RecordedEvent: An instance of the RecordedEvent class for managing recorded events.

        Example:
        ```
        adb_plus = AdbControlPlus(adb_path="/path/to/adb", device_serial="123456")
        results = adb_plus.plus_record_and_save_to_sdcard(
            tmpfolder_device='/sdcard/clickevent2',
            tmpfolder_local=None,
            device='event4',
            print_output=True,
            add_closing_command=True,
            clusterevents=16,
        )
        results()
        ```

        """
        self.sh_mkdir(tmpfolder_device)
        tmpfolder_device = "/" + tmpfolder_device.strip("/") + "/"
        if not tmpfolder_local:
            tmpfolder_local = os.path.normpath(
                os.path.join(os.getcwd(), str(time.time()).split(".")[0])
            )
            os.makedirs(tmpfolder_local, exist_ok=True)

        results = GeteventPlayBack(
            adb_path=self.adb_path,
            device=f"/dev/input/{device}",
            device_serial=self.device_serial,
            print_output=print_output,
            tmpfolder_device=tmpfolder_device,
            tempfolder_hdd=tmpfolder_local,
            add_closing_command=add_closing_command,
            clusterevents=clusterevents,
        ).start_recording()

        commandfile = f"{tmpfolder_device}cmd.txt"
        self.sh_create_file_with_content(results["adbcommand"], commandfile)

        return RecordedEvent(self.execute_sh_command, commandfile)

    def plus_start_logcat(self, logfile=None, devices=None):
        r"""
        Start capturing logcat output from specified devices.

        Parameters:
        - logfile (str): The path to the log file. If None, a default file is created in the current working directory.
        - devices (list): List of device serials to capture logcat from. If None, all connected devices are used.

        Returns:
        subprocess.Popen: A subprocess representing the running logcat process.

        Example:
        ```
        adb_plus = AdbControlPlus(adb_path="/path/to/adb", device_serial="123456")
        logcat_process = adb_plus.plus_start_logcat(logfile="/path/to/logcat.csv", devices=["device1", "device2"])
        ```

        Note:
        - Logcat output is captured in CSV format.
        - The logcat process is started in a new console window.

        """

        if not logfile:
            logfile = os.path.join(
                os.getcwd(), "ANDROIDLOGCAT", str(time.time()) + ".csv"
            )
        print(logfile)
        touch(logfile)
        if not devices:
            devices = [x[0] for x in self.get_all_devices() if "offline" not in x]
        command = f"from logcatdevices import start_logcat; _ = start_logcat(adb_path=r'{self.adb_path}', csv_output={repr(logfile)}, device_serials={repr(devices)}, print_stdout=True, clear_log=True, ljustserial=16)"
        return subprocess.Popen(
            [sys.executable, "-c", command], creationflags=subprocess.CREATE_NEW_CONSOLE
        )

    def plus_start_fast_screenshot_iter(
        self,
        bitrate="20M",
        screenshotbuffer=10,
        go_idle=0,
    ):
        r"""
        Start capturing fast screenshots from the connected device in an iterator.

        Parameters:
        - bitrate (str): The bitrate for video encoding, e.g., "20M" for 20 megabits per second.
        - screenshotbuffer (int): Number of screenshots to buffer before yielding the next one.
        - go_idle (int): Time to wait before the device goes idle (default is 0).

        Yields:
        numpy.ndarray: A NumPy array representing the screenshot image.

        Example:
        ```
        adb_plus = AdbControlPlus(adb_path="/path/to/adb", device_serial="123456")
        for image in adb_plus.plus_start_fast_screenshot_iter(bitrate="10M", screenshotbuffer=5, go_idle=2):
            cv2.imshow("CV2 WINDOW", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
        ```

        Note:
        - The iterator yields fast screenshots at the specified time interval and bitrate.
        - Displaying the images using OpenCV is shown in the example.

        """
        self._get_height_width()

        with AdbFastScreenshots(
            adb_path=self.adb_path,
            device_serial=self.device_serial,
            time_interval=179,
            width=self.device_width,
            height=self.device_height,
            bitrate=bitrate,
            use_busybox=self.use_busybox,
            connect_to_device=False,
            screenshotbuffer=screenshotbuffer,
            go_idle=go_idle,
        ) as adbscreen:
            for image in adbscreen:
                yield image

    def plus_uidump(
        self,
        timeout=60,
        with_screenshot=True,
        screenshot=None,
        nice=False,
        su=False,
        with_sendevent=True,
    ):
        r"""
        Perform a UI dump using uiautomator and enhance the resulting DataFrame with additional methods.

        Parameters:
        - timeout (int): Maximum time to wait for the UI dump operation (default is 60 seconds).
        - with_screenshot (bool): Include screenshots in the DataFrame (default is True).
        - screenshot (numpy.ndarray): Pre-captured screenshot to be included in the DataFrame.
        - nice (bool): Use 'nice' for UI dump execution, improving performance (default is False). Important: nice needs su!
        - su (bool): Use 'su' for UI dump execution, granting superuser privileges (default is False).
        - with_sendevent (bool): Include sendevent methods in the DataFrame (default is True).

        Returns:
        DataFrameWithMeta: A DataFrame with added methods for interacting with the UI dump.

        Example:
        ```
        adb_plus = AdbControlPlus(adb_path="/path/to/adb", device_serial="123456")
        ui_dump_df = adb_plus.plus_uidump(timeout=30, with_screenshot=True, nice=True, su=True)
        # Access methods for interacting with the UI dump
        ui_dump_df.aa_click(100, 200)
        ui_dump_df.aa_long_click(300, 400)
        ```

        Note:
        - The resulting DataFrame includes additional methods for UI interaction.
        - Methods like `aa_click`, `aa_long_click`, etc., are added for convenient input.
        - You can use the provided DataFrame for interacting with the UI dump.

        """
        with_input_tap = True
        df = self.uiautomator_nice20(timeout=timeout, nice=nice, su=su, as_pandas=True)
        df.columns = [f"aa_{x}" for x in df.columns]
        df = self._add_df_methods(
            df, with_screenshot, screenshot, with_input_tap, with_sendevent
        )

        dfx = DataFrameWithMeta(
            df,
            adb_instance=self,
        )
        dfx.attrs["adb_instance"] = self
        return dfx

    def plus_uidump_with_freeze(
        self,
        procregex_for_lsof,
        timeout=60,
        with_screenshot=True,
        screenshot=None,
        with_sendevent=True,
    ):
        r"""
        Perform a UI dump with freezing specified processes and enhance the resulting
        DataFrame with additional methods.

        Parameters:
        - procregex_for_lsof (str or bytes): Regular expression pattern or bytes to match processes for freezing.
        - timeout (int): Maximum time to wait for the UI dump operation (default is 60 seconds).
        - with_screenshot (bool): Include screenshots in the DataFrame (default is True).
        - screenshot (numpy.ndarray): Pre-captured screenshot to be included in the DataFrame.
        - with_sendevent (bool): Include sendevent methods in the DataFrame (default is True).

        Returns:
        DataFrameWithMeta: A DataFrame with added methods for interacting with the UI dump,
        considering frozen processes.

        Example:
        ```
        adb_plus = AdbControlPlus(adb_path="/path/to/adb", device_serial="123456")
        # Freeze processes matching the given regex during UI dump
        ui_dump_df = adb_plus.plus_uidump_with_freeze(
            procregex_for_lsof="com.example.app",
            timeout=30,
            with_screenshot=True,
            with_sendevent=True,
        )
        # Access methods for interacting with the UI dump considering frozen processes
        ui_dump_df.aa_click(100, 200)
        ui_dump_df.aa_long_click(300, 400)
        ```

        Note:
        - Processes matching the provided regular expression will be frozen during the UI dump.
        - The resulting DataFrame includes additional methods for UI interaction.
        - Methods like `aa_click`, `aa_long_click`, etc., are added for convenient input.
        - You can use the provided DataFrame for interacting with the UI dump.

        """
        if isinstance(procregex_for_lsof, str):
            procregex_for_lsof = procregex_for_lsof.encode()

        if isinstance(procregex_for_lsof, bytes):
            procregex_for_lsof = re.compile(procregex_for_lsof)

        procpids = []

        procresults = [
            x
            for x in self.sh_get_details_with_lsof(su=True)[0]
            if procregex_for_lsof.search(x)
        ]
        procr = defaultdict(list)
        for p in [x.split(maxsplit=2) for x in procresults]:
            procr[f"{int(p[1])}_pid"].append([p[2]])
            namecol = f"{int(p[1])}_name"
            if namecol not in procr:
                procr[namecol] = p[0]
        for k, v in procr.items():
            if "pid" in k:
                onlypid = k.split("_")[0]
                procpid = int(onlypid)
                self.sh_freeze_proc(procpid)
                procpids.append(procpid)
        df = self.plus_uidump(
            timeout=timeout,
            with_screenshot=with_screenshot,
            screenshot=screenshot,
            nice=True,
            su=True,
            with_sendevent=with_sendevent,
        )
        for p in procpids:
            self.sh_unfreeze_proc(p)

        dfx = DataFrameWithMeta(
            df,
            adb_instance=self,
        )
        dfx.attrs["adb_instance"] = self
        return dfx

    def _add_df_methods(
        self, df, with_screenshot, screenshot, with_input_tap, with_sendevent
    ):
        r"""
        Enhance a DataFrame with additional methods for interacting with the UI elements and events.

        Parameters:
        - df (DataFrameWithMeta): The input DataFrame with UI elements and events.
        - with_screenshot (bool): Include screenshots in the DataFrame (default is True).
        - screenshot (numpy.ndarray): Pre-captured screenshot to be included in the DataFrame.
        - with_input_tap (bool): Include input tap methods in the DataFrame (default is True).
        - with_sendevent (bool): Include sendevent methods in the DataFrame (default is True).

        Returns:
        DataFrameWithMeta: A DataFrame with added methods for UI interaction and event handling.

        Example:
        ```
        adb_plus = AdbControlPlus(adb_path="/path/to/adb", device_serial="123456")
        # Perform a UI dump and enhance the resulting DataFrame with additional methods
        ui_dump_df = adb_plus.plus_uidump(with_input_tap=True, with_sendevent=True)
        # Enhance the DataFrame with input tap and sendevent methods
        enhanced_df = adb_plus._add_df_methods(
            ui_dump_df,
            with_screenshot=True,
            screenshot=ui_dump_df.aa_screenshot.iloc[0],
            with_input_tap=True,
            with_sendevent=True,
        )
        # Access methods for interacting with the UI elements and events
        enhanced_df.ff_input_tap(100, 200)
        enhanced_df.ff_sendevent_touch(300, 400, inputdev="event0", inputdevmax=255)
        ```

        Note:
        - The resulting DataFrame includes additional methods for UI interaction and event handling.
        - Methods like `ff_input_tap`, `ff_sendevent_touch`, etc., are added for convenient interaction.
        - You can use the provided DataFrame for interacting with the UI elements and handling events.

        """
        if with_screenshot:
            if not is_truthy(screenshot):
                screenshot = self.plus_screenshot_as_np()
            df.loc[df.aa_area > 0, "aa_screenshot"] = df.loc[df.aa_area > 0].apply(
                lambda x: cropimage(
                    screenshot, (x.aa_start_x, x.aa_start_y, x.aa_end_x, x.aa_end_y)
                ),
                axis=1,
            )
        dfx = DataFrameWithMeta(
            _add_click_methods(
                DataFrameWithMeta(
                    df,
                    adb_instance=self,
                ),
                with_input_tap=with_input_tap,
                with_sendevent=with_sendevent,
                column_input_tap="ff_input_tap",
                column_x="aa_center_x",
                column_y="aa_center_y",
                sendevent_prefix="ff_sendevent_",
            ),
            adb_instance=self,
        )

        dfx.attrs["adb_instance"] = self
        return dfx

    def _get_height_width(self):
        if not self.device_width or not self.device_height:
            self.device_width, self.device_height = self.sh_get_wm_size()
        return self.device_width, self.device_height

    def plus_screenshot_as_np(self):
        r"""
        Capture and decode the device's screenshot as a NumPy array.

        Returns:
        numpy.ndarray: A NumPy array representing the screenshot in BGR color format.

        Example:
        ```
        adb_plus = AdbControlPlus(adb_path="/path/to/adb", device_serial="123456")
        screenshot_np = adb_plus.plus_screenshot_as_np()
        # Use the NumPy array for image processing or display
        cv2.imshow("Device Screenshot", screenshot_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ```

        Note:
        - This method uses the `sh_screencap_png` method to capture the device's screenshot as a PNG.
        - The PNG data is then decoded using OpenCV (`cv2.imdecode`) to obtain the NumPy array.
        - The resulting NumPy array represents the screenshot in BGR color format, suitable for image processing.

        """
        return cv2.imdecode(
            np.frombuffer(
                self.sh_screencap_png(
                    correct_newlines=False,
                    use_eval=True,
                    eval_timeout=3,
                ),
                np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )

    def plus_activity_elements_dump(
        self, with_screenshot=True, screenshot=None, with_sendevent=False
    ):
        r"""
        Dump the activity elements and create a DataFrame with metadata.

        Args:
        with_screenshot (bool, optional): Include screenshots for visible elements. Defaults to True.
        screenshot (numpy.ndarray, optional): Pre-captured screenshot in BGR format. Defaults to None.
        with_sendevent (bool, optional): Include sendevent click functions (needs SU!). Defaults to False.

        Returns:
        DataFrameWithMeta: A DataFrame with metadata containing information about activity elements.

        Example:
        ```
        adb_plus = AdbControlPlus(adb_path="/path/to/adb", device_serial="123456")
        elements_df = adb_plus.plus_activity_elements_dump()
        # Access DataFrame columns and perform analysis
        print(elements_df["aa_text"])
        ```

        Note:
        - This method retrieves activity elements using the `get_all_activity_elements` method.
        - The elements are organized into a DataFrame with metadata, including parent-child relationships.
        - Optional features like screenshots, input tap methods, and sendevent methods can be included.
        - The resulting DataFrame is an instance of DataFrameWithMeta for additional functionality.

        """
        with_input_tap = True
        mappingdict = {}
        i = 0
        chidi = defaultdict(list)
        acti = self.get_all_activity_elements(as_pandas=False)
        for ini, act in enumerate(acti):
            for ini2, act2 in enumerate(act):
                for ini3, act3 in enumerate(act2):
                    tudi = tuple(act3.items())[:-2]
                    if tudi not in mappingdict:
                        mappingdict[tudi] = i
                        i += 1
                for ini3, act3 in enumerate(act2[:-1]):
                    chidi[tuple(act2[-1].items())[:-2]].append(
                        mappingdict.get(tuple(act3.items())[:-2])
                    )

        df = DataFrameWithMeta(
            [[h[1] for h in q] for q in chidi.keys()],
            adb_instance=self,
            columns=list(
                flatten_everything([[h[0] for h in q] for q in list(chidi.keys())[:1]])
            ),
        )
        df["aa_parents"] = [tuple(r) for r in chidi.values()]
        df.columns = [f"aa_{x.lower()}" for x in df.columns]
        df["aa_element_index"] = df.index.__array__().copy()
        dfx = DataFrameWithMeta(
            self._add_df_methods(
                df, with_screenshot, screenshot, with_input_tap, with_sendevent
            ),
            adb_instance=self,
        )

        dfx.attrs["adb_instance"] = self
        return dfx
