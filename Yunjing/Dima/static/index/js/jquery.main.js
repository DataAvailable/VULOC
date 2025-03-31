// page init
jQuery(function(){
	"use strict";

	initIsoTop();
	initAnchors();
	initCounter();
	initbackTop();
	initLightbox();
	initAddClass();
	new WOW().init();
	initTextRotator();
	initSlickSlider();
	initProgressBar();
	initVegasSlider();
	initTextRotator2();
	initTextRotator3();
	initStickyHeader();
});
jQuery(window).on('load', function() {
	"use strict";

	initPreLoader();
	new WOW().init();
});

// PreLoader init
function initPreLoader() {
	"use strict";

	jQuery('#pre-loader').delay(1000).fadeOut();
}

// TextRotator init
function initTextRotator() {
	"use strict";

	jQuery("#js-rotating").Morphext({
		// The [in] animation type. Refer to Animate.css for a list of available animations.
		animation: "bounceIn",
		// An array of phrases to rotate are created based on this separator. Change it if you wish to separate the phrases differently (e.g. So Simple | Very Doge | Much Wow | Such Cool).
		separator: ",",
		// The delay between the changing of each phrase in milliseconds.
		speed: 2000,
		complete: function () {
		    // Called after the entrance animation is executed.
		}
	});
}

// TextRotator2 init
function initTextRotator2() {
	"use strict";

	jQuery('#rotating2').textillate({
		selector: '.rotating-hold',

		// enable looping
		loop: true,

		// sets the minimum display time for each text before it is replaced
		minDisplayTime: 2000,

		// sets the initial delay before starting the animation
		// (note that depending on the in effect you may need to manually apply
		// visibility: hidden to the element before running this plugin)
		initialDelay: 0,

		// set whether or not to automatically start animating
		autoStart: true,

		// custom set of 'in' effects. This effects whether or not the
		// character is shown/hidden before or after an animation
		inEffects: [],

		// custom set of 'out' effects
		outEffects: [ 'hinge' ],

		// in animation settings
		in: {
			// set the effect name
			effect: 'fadeInLeftBig',

			// set the delay factor applied to each consecutive character
			delayScale: 1.5,

			// set the delay between each character
			delay: 50,

			// set to true to animate all the characters at the same time
			sync: false,

			// randomize the character sequence
			// (note that shuffle doesn't make sense with sync = true)
			shuffle: false,

			// reverse the character sequence
			// (note that reverse doesn't make sense with sync = true)
			reverse: false
		},
		out: {
			effect: 'hinge',
			delayScale: 1.5,
			delay: 50,
			sync: false,
			shuffle: false,
			reverse: false,
		},
		type: 'char'
	});
}

// TextRotator2 init
function initTextRotator3() {
	"use strict";

	jQuery("#rotating3").typed({
		strings: ["the best", "the Simple", "the Doge", "the Cool"],
		loop: true,
		typeSpeed: 100
	});
}

// modal popup init
function initLightbox() {
	"use strict";

	jQuery('a.lightbox, a[rel*="lightbox"]').fancybox({
		helpers: {
			overlay: {
				css: {
					background: 'rgba(0, 0, 0, 0.65)'
				}
			}
		},
		afterLoad: function(current, previous) {
			// handle custom close button in inline modal
			if(current.href.indexOf('#') === 0) {
				jQuery(current.href).find('a.close').off('click.fb').on('click.fb', function(e){
					e.preventDefault();
					jQuery.fancybox.close();
				});
			}
		},
		padding: 0
	});
}

// Progress Bar init
function initProgressBar() {
	"use strict";

	if( jQuery(".progress-bar").length != '' ){
		var waypoint = new Waypoint({
			element: document.getElementById('progress-bar'),
			handler: function(direction) {
				console.log('Scrolled to waypoint!');
				jQuery('.progress-bar li').each(function() {
						var widthBar = jQuery(this).find('.over').attr('data-percent');
						jQuery(this).find('.over').animate({
							'width': widthBar
						});
				});
			},
			offset: '80%'
		});
	}
}

// Vegas Slider init
function initVegasSlider() {
	"use strict";

  jQuery("#bgvid").vegas({
      slides: [
        {   src: 'http://placehold.it/1920x1080',
            video: {
                src: [
                    'video/polina.webm',
                    'video/polina.mp4'
                ],
                loop: true,
                timer: false,
                mute: true
            }
        }
    ]
  });
}

// Slick Slider init
function initSlickSlider() {
	"use strict";

	jQuery('.test-slider').slick({
		dots: true,
		speed: 800,
		infinite: true,
		slidesToShow: 1,
		adaptiveHeight: true,
		autoplay: true,
		arrows: false,
		autoplaySpeed: 4000
	});
	jQuery('.client-slider').slick({
		speed: 800,
		dots: false,
		infinite: true,
		autoplay: true,
		slidesToShow: 1,
		adaptiveHeight: true,
		autoplaySpeed: 4000
	});
	jQuery('.top-slider').slick({
		dots: true,
		speed: 800,
		arrows: false,
		infinite: true,
		autoplay: true,
		slidesToShow: 1,
		autoplaySpeed: 4000,
		adaptiveHeight: true
	});
	jQuery('.top-slider-fade').slick({
		dots: true,
		speed: 800,
		fade: true,
		arrows: false,
		infinite: true,
		autoplay: true,
		slidesToShow: 1,
		autoplaySpeed: 4000,
		adaptiveHeight: true
	});
	jQuery('.top-slider-fadeani').slick({
		dots: true,
		speed: 800,
		fade: true,
		arrows: false,
		infinite: true,
		autoplay: true,
		slidesToShow: 1,
		autoplaySpeed: 4000,
		adaptiveHeight: true

	}).on('afterChange', function(event, slick, currentSlide, nextSlide){jQuery(this).find('.slide').eq(currentSlide).find('.animated').addClass('flipInX').removeClass('hide');}).on('beforeChange', function(event, slick, currentSlide, nextSlide){jQuery(this).find('.slide').eq(currentSlide).find('.animated').removeClass('flipInX').addClass('hide');});
}

// Add Class  init
function initAddClass() {
	"use strict";

	jQuery('.popup-opener').on( "click", function(e){
		e.preventDefault();
		jQuery("body").toggleClass("popup-active");
	});
	jQuery('.navbar .smooth').on( "click", function(){
		if (jQuery(".navbar-collapse.collapse").hasClass("in")) {
			setTimeout(function() { 
			jQuery(".navbar-toggle").trigger('click')}, 800);
		}
	});
}

// sticky header init
function initStickyHeader() {
	"use strict";

	var win = jQuery(window),
		stickyClass = 'sticky';

	jQuery('#header').each(function() {
		var header = jQuery(this);
		var headerOffset = header.offset().top || 200;
		var flag = true;

		jQuery(this).css('height' , jQuery(this).innerHeight());

		function scrollHandler() {
			if (win.scrollTop() > headerOffset) {
				if (flag){
					flag = false;
					header.addClass(stickyClass);
				}
			} else {
				if (!flag) {
					flag = true;
					header.removeClass(stickyClass);
				}
			}
		}

		scrollHandler();
		win.on('scroll resize orientationchange', scrollHandler);
	});
}

// Counter init
function initCounter() {
	"use strict";

	jQuery('.counter').counterUp({
		delay: 10,
		time: 2000
	});
}

// IsoTop init
function initIsoTop() {
	"use strict";

	// Isotope init
	var isotopeHolder = jQuery('.isoto-holder'),
		win = jQuery(window);
	jQuery('.isoto-filter a').on( "click", function(e){
		e.preventDefault();
		
		jQuery('.isoto-filter li').removeClass('active');
		jQuery(this).parent('li').addClass('active');
		var selector = jQuery(this).attr('data-filter');
		isotopeHolder.isotope({ filter: selector });
	});
	jQuery('.isoto-holder').isotope({
		itemSelector: '.coll',
		transitionDuration: '0.6s',
		masonry: {
			columnWidth: '.coll'
		}
	});
}

// smooth anchor links init
function initAnchors() {
	"use strict";

	new SmoothScroll({
		anchorLinks: 'a.smooth[href^="#"]:not([href="#"])',
		extraOffset: 80,
		activeClasses: 'parent'
	});
}


// backtop init
function initbackTop() {
	"use strict";

	var jQuerybackToTop = jQuery("#back-top");
	jQuery(window).on('scroll', function() {
		if (jQuery(this).scrollTop() > 100) {
			jQuerybackToTop.addClass('active');
		} else {
			jQuerybackToTop.removeClass('active');
		}
	});
	jQuerybackToTop.on('click', function(e) {
		jQuery("html, body").animate({scrollTop: 0}, 900);
	});
}
