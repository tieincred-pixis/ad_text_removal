import boto3
from PIL import Image
from io import BytesIO


class S3Utils:
    def __init__(self) -> None:
        self.bucket_name = "rnd-creative-diffusion"
        self.access_key = "AKIAYK2LLOFHBGXWRJMF"
        self.secret_key = "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"
        self.region_name = "ap-south-1"

        self.s3 = boto3.resource(
            's3',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region_name)

    def read_image_from_s3(self, s3_path):
        """Load image file from s3.

        Parameters
        ----------
        s3_path : string
            Path in s3

        Returns
        -------
        PIL image
            Image
        """
        bucket = self.s3.Bucket(self.bucket_name)
        object = bucket.Object(s3_path)
        response = object.get()
        file_stream = response['Body']
        im = Image.open(file_stream)
        return im

    def read_from_bucket_folder(self, folder):
        """
        This functions list all files in s3 bucket.
        :return: None
        """

        bucket = self.s3.Bucket(self.bucket_name)
        results = []
        for object_summary in bucket.objects.filter(Prefix=folder):
            results.append(object_summary.key)
        return results

    def write_image_to_s3(self, im, upload_path):
        """Write an image array into S3 bucket

        Parameters
        ----------
        img_array: PIL image
            image
        upload_path : string
            Upload Path in s3

        Returns
        -------
        None
        """
        bucket = self.s3.Bucket(self.bucket_name)
        object = bucket.Object(self.access_key)
        file_stream = BytesIO()
        # im = Image.fromarray(img_array)
        try:
            im.save(file_stream, format='jpeg')
        except:
            im.save(file_stream, format='png')
        object.put(Body=file_stream.getvalue(), Key=upload_path)
