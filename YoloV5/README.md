<h1> YOLO V5</h1>
<p> Bu kalsör YoloV5 nöral ağı için kullanılan tespit algoritmasının çalışma mantığını anlamak için yaptığım testler ve 
çalışmaları barındırıyor. Bu kapsamda yazdığım kodların ve yaptığım ekleme-çıkarmaların hepsini kalasörde bulabilirsiniz.
</p>

<h3> TEST 1 </h3>
<p>YoloV5 için yaygın olarak kullanılan tespit algoritması, main_class.py, sınıf yapısıda olduğu için ilk başta anlaması 
daha doğrusu satır satır incelemesi zor olacağı için bu algoritmayı sınıf yapısından çıkartarak resimler için ve videolar
için çalışır hale getiridim. Bu sayede kodu satır satır takip ederek hangi satırın ne yaptığını daha kolay anlayabildim.
Bu kısımda kabaca tespit algoritmasının çoğunun ne işe yaradığını anlayabilirisiniz. Kendi notlarım klasördeki pdf'nin
içerisinde var.</p>

<h3> TEST 2 </h3>
<p>Resimler için tespit yapan algoritmayı jupyter notebook formatına taşıdıktan sonra satır satır çalıştırarak daha 
detaylı bir incelem yapabildim. Burada "results" değişkeninin iç yapısı dışında her şey yerine oturdu. Results yapısını
daha iyi anlamak için internetten ufak bir destek aldım, Pytorch'un resmi sitesinde bulunan bir çok method'u test ederken
results değişkenini de daha detaylı inceledim. YoloV5 ile kullanılabilecek ekstra methodlarda pdf'de blunabilir.</p>

<h3> TEST 3 </h3>
<p>Artık algoritmanın nasıl çalıştığını çözdüğümüze göre artık algoritmaya kendi eklemelerimizi yapabiliriz. Bu kapsamda 
ilk olarak tespit edilen objelerin merkez noktalarını döndüren bir fonksiyon ekledim. Sonrasında ise bu merkez noktanın 
ekranın neresinde olduğunu (sağ-sol-orta-yukarı-aşağı) ve etiketinin ne olduğunu döndüren yeni fonksiyonlar ekledim.</p>