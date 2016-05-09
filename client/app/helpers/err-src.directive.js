(function() {
  'use strict';

  angular
    .module('bubbleApp.helpers')
    .directive('errSrcDirective', errSrcDirective);

  function errSrcDirective() {
    return {
      link: function(scope, iElement, iAttrs) {
        iElement.bind('error', function() {
          angular.element(this).attr("src", iAttrs.errSrc);
        });
      }
    };
  }
})();