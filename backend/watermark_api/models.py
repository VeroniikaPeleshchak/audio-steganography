from django.db import models
import uuid

class AudioTrack(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    original_audio = models.FileField(upload_to='uploads/originals/')
    
    watermarked_audio = models.FileField(upload_to='uploads/watermarked/', null=True, blank=True)
    
    watermark_data = models.CharField(max_length=255, help_text="Текст або ідентифікатор водяного знаку")
    
    algorithm = models.CharField(max_length=50, default='dwt_svd')

    is_ecc = models.BooleanField(default=False, help_text="Чи активовано захист ECC")
    
    snr = models.FloatField(null=True, blank=True)

    psnr = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Track {self.id} [{self.algorithm}] - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
class ExpertiseResult(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    track = models.ForeignKey(AudioTrack, on_delete=models.CASCADE, related_name='results')

    attack_type = models.CharField(max_length=100) 

    attack_params = models.CharField(max_length=100) 
    
    extracted_text = models.TextField()

    ber = models.FloatField()
    
    ncc = models.FloatField()

    snr = models.FloatField(null=True, blank=True)

    psnr = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)